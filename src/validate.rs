//! DNSSEC validation.
//!
//! **This module is experimental and likely to change significantly.**
#![cfg(feature = "unstable-validate")]
#![cfg_attr(docsrs, doc(cfg(feature = "unstable-validate")))]

use crate::base::cmp::CanonicalOrd;
use crate::base::iana::{DigestAlg, SecAlg};
use crate::base::name::Name;
use crate::base::name::ToName;
use crate::base::rdata::{ComposeRecordData, RecordData};
use crate::base::record::Record;
use crate::base::scan::IterScanner;
use crate::base::wire::{Compose, Composer};
use crate::rdata::{Dnskey, Rrsig};
use bytes::Bytes;
use octseq::builder::with_infallible;
use ring::{digest, signature};
use std::boxed::Box;
use std::vec::Vec;
use std::{error, fmt};

/// A generic public key.
#[derive(Clone, Debug)]
pub enum RawPublicKey {
    /// An RSA/SHA-1 public key.
    RsaSha1(RsaPublicKey),

    /// An RSA/SHA-1 with NSEC3 public key.
    RsaSha1Nsec3Sha1(RsaPublicKey),

    /// An RSA/SHA-256 public key.
    RsaSha256(RsaPublicKey),

    /// An RSA/SHA-512 public key.
    RsaSha512(RsaPublicKey),

    /// An ECDSA P-256/SHA-256 public key.
    ///
    /// The public key is stored in uncompressed format:
    ///
    /// - A single byte containing the value 0x04.
    /// - The encoding of the `x` coordinate (32 bytes).
    /// - The encoding of the `y` coordinate (32 bytes).
    EcdsaP256Sha256(Box<[u8; 65]>),

    /// An ECDSA P-384/SHA-384 public key.
    ///
    /// The public key is stored in uncompressed format:
    ///
    /// - A single byte containing the value 0x04.
    /// - The encoding of the `x` coordinate (48 bytes).
    /// - The encoding of the `y` coordinate (48 bytes).
    EcdsaP384Sha384(Box<[u8; 97]>),

    /// An Ed25519 public key.
    ///
    /// The public key is a 32-byte encoding of the public point.
    Ed25519(Box<[u8; 32]>),

    /// An Ed448 public key.
    ///
    /// The public key is a 57-byte encoding of the public point.
    Ed448(Box<[u8; 57]>),
}

impl RawPublicKey {
    /// The algorithm used by this key.
    pub fn algorithm(&self) -> SecAlg {
        match self {
            Self::RsaSha1(_) => SecAlg::RSASHA1,
            Self::RsaSha1Nsec3Sha1(_) => SecAlg::RSASHA1_NSEC3_SHA1,
            Self::RsaSha256(_) => SecAlg::RSASHA256,
            Self::RsaSha512(_) => SecAlg::RSASHA512,
            Self::EcdsaP256Sha256(_) => SecAlg::ECDSAP256SHA256,
            Self::EcdsaP384Sha384(_) => SecAlg::ECDSAP384SHA384,
            Self::Ed25519(_) => SecAlg::ED25519,
            Self::Ed448(_) => SecAlg::ED448,
        }
    }
}

impl RawPublicKey {
    /// Parse a public key as stored in a DNSKEY record.
    pub fn from_dnskey(
        algorithm: SecAlg,
        data: &[u8],
    ) -> Result<Self, FromDnskeyError> {
        match algorithm {
            SecAlg::RSASHA1 => {
                RsaPublicKey::from_dnskey(data).map(Self::RsaSha1)
            }
            SecAlg::RSASHA1_NSEC3_SHA1 => {
                RsaPublicKey::from_dnskey(data).map(Self::RsaSha1Nsec3Sha1)
            }
            SecAlg::RSASHA256 => {
                RsaPublicKey::from_dnskey(data).map(Self::RsaSha256)
            }
            SecAlg::RSASHA512 => {
                RsaPublicKey::from_dnskey(data).map(Self::RsaSha512)
            }

            SecAlg::ECDSAP256SHA256 => {
                let mut key = Box::new([0u8; 65]);
                if key.len() == 1 + data.len() {
                    key[0] = 0x04;
                    key[1..].copy_from_slice(data);
                    Ok(Self::EcdsaP256Sha256(key))
                } else {
                    Err(FromDnskeyError::InvalidKey)
                }
            }
            SecAlg::ECDSAP384SHA384 => {
                let mut key = Box::new([0u8; 97]);
                if key.len() == 1 + data.len() {
                    key[0] = 0x04;
                    key[1..].copy_from_slice(data);
                    Ok(Self::EcdsaP384Sha384(key))
                } else {
                    Err(FromDnskeyError::InvalidKey)
                }
            }

            SecAlg::ED25519 => Box::<[u8]>::from(data)
                .try_into()
                .map(Self::Ed25519)
                .map_err(|_| FromDnskeyError::InvalidKey),
            SecAlg::ED448 => Box::<[u8]>::from(data)
                .try_into()
                .map(Self::Ed448)
                .map_err(|_| FromDnskeyError::InvalidKey),

            _ => Err(FromDnskeyError::UnsupportedAlgorithm),
        }
    }

    /// Parse a public key from a DNSKEY record in presentation format.
    ///
    /// This format is popularized for storing alongside private keys by the
    /// BIND name server.  This function is convenient for loading such keys.
    ///
    /// The text should consist of a single line of the following format (each
    /// field is separated by a non-zero number of ASCII spaces):
    ///
    /// ```text
    /// <domain-name> <record-class> DNSKEY <record-data> [<comment>]
    /// ```
    ///
    /// Where `<record-data>` consists of the following fields:
    ///
    /// ```text
    /// <flags> <protocol> <algorithm> <encoded-public-key>
    /// ```
    ///
    /// The first three fields are simple integers, while the last field is
    /// Base64 encoded data (with or without padding).  The [`from_dnskey()`]
    /// and [`to_dnskey()`] read from and serialize to the Base64-decoded data
    /// format.
    ///
    /// [`from_dnskey()`]: Self::from_dnskey()
    /// [`to_dnskey()`]: Self::to_dnskey()
    ///
    /// The `<comment>` is any text starting with an ASCII semicolon.
    pub fn parse_dnskey_text(
        dnskey: &str,
    ) -> Result<Self, FromDnskeyTextError> {
        // Ensure there is a single line in the input.
        let (line, rest) = dnskey.split_once('\n').unwrap_or((dnskey, ""));
        if !rest.trim().is_empty() {
            return Err(FromDnskeyTextError::Misformatted);
        }

        // Strip away any semicolon from the line.
        let (line, _) = line.split_once(';').unwrap_or((line, ""));

        // Ensure the record header looks reasonable.
        let mut words = line.split_ascii_whitespace().skip(2);
        if !words.next().unwrap_or("").eq_ignore_ascii_case("DNSKEY") {
            return Err(FromDnskeyTextError::Misformatted);
        }

        // Parse the DNSKEY record data.
        let mut data = IterScanner::new(words);
        let dnskey: Dnskey<Vec<u8>> = Dnskey::scan(&mut data)
            .map_err(|_| FromDnskeyTextError::Misformatted)?;
        println!("importing {:?}", dnskey);
        Self::from_dnskey(dnskey.algorithm(), dnskey.public_key().as_slice())
            .map_err(FromDnskeyTextError::FromDnskey)
    }

    /// Serialize this public key as stored in a DNSKEY record.
    pub fn to_dnskey(&self) -> Box<[u8]> {
        match self {
            Self::RsaSha1(k)
            | Self::RsaSha1Nsec3Sha1(k)
            | Self::RsaSha256(k)
            | Self::RsaSha512(k) => k.to_dnskey(),

            // From my reading of RFC 6605, the marker byte is not included.
            Self::EcdsaP256Sha256(k) => k[1..].into(),
            Self::EcdsaP384Sha384(k) => k[1..].into(),

            Self::Ed25519(k) => k.as_slice().into(),
            Self::Ed448(k) => k.as_slice().into(),
        }
    }
}

impl PartialEq for RawPublicKey {
    fn eq(&self, other: &Self) -> bool {
        use ring::constant_time::verify_slices_are_equal;

        match (self, other) {
            (Self::RsaSha1(a), Self::RsaSha1(b)) => a == b,
            (Self::RsaSha1Nsec3Sha1(a), Self::RsaSha1Nsec3Sha1(b)) => a == b,
            (Self::RsaSha256(a), Self::RsaSha256(b)) => a == b,
            (Self::RsaSha512(a), Self::RsaSha512(b)) => a == b,
            (Self::EcdsaP256Sha256(a), Self::EcdsaP256Sha256(b)) => {
                verify_slices_are_equal(&**a, &**b).is_ok()
            }
            (Self::EcdsaP384Sha384(a), Self::EcdsaP384Sha384(b)) => {
                verify_slices_are_equal(&**a, &**b).is_ok()
            }
            (Self::Ed25519(a), Self::Ed25519(b)) => {
                verify_slices_are_equal(&**a, &**b).is_ok()
            }
            (Self::Ed448(a), Self::Ed448(b)) => {
                verify_slices_are_equal(&**a, &**b).is_ok()
            }
            _ => false,
        }
    }
}

/// A generic RSA public key.
///
/// All fields here are arbitrary-precision integers in big-endian format,
/// without any leading zero bytes.
#[derive(Clone, Debug)]
pub struct RsaPublicKey {
    /// The public modulus.
    pub n: Box<[u8]>,

    /// The public exponent.
    pub e: Box<[u8]>,
}

impl RsaPublicKey {
    /// Parse an RSA public key as stored in a DNSKEY record.
    pub fn from_dnskey(data: &[u8]) -> Result<Self, FromDnskeyError> {
        if data.len() < 3 {
            return Err(FromDnskeyError::InvalidKey);
        }

        // The exponent length is encoded as 1 or 3 bytes.
        let (exp_len, off) = if data[0] != 0 {
            (data[0] as usize, 1)
        } else if data[1..3] != [0, 0] {
            // NOTE: Even though this is the extended encoding of the length,
            // a user could choose to put a length less than 256 over here.
            let exp_len = u16::from_be_bytes(data[1..3].try_into().unwrap());
            (exp_len as usize, 3)
        } else {
            // The extended encoding of the length just held a zero value.
            return Err(FromDnskeyError::InvalidKey);
        };

        // NOTE: off <= 3 so is safe to index up to.
        let e = data[off..]
            .get(..exp_len)
            .ok_or(FromDnskeyError::InvalidKey)?
            .into();

        // NOTE: The previous statement indexed up to 'exp_len'.
        let n = data[off + exp_len..].into();

        Ok(Self { n, e })
    }

    /// Serialize this public key as stored in a DNSKEY record.
    pub fn to_dnskey(&self) -> Box<[u8]> {
        let mut key = Vec::new();

        // Encode the exponent length.
        if let Ok(exp_len) = u8::try_from(self.e.len()) {
            key.reserve_exact(1 + self.e.len() + self.n.len());
            key.push(exp_len);
        } else if let Ok(exp_len) = u16::try_from(self.e.len()) {
            key.reserve_exact(3 + self.e.len() + self.n.len());
            key.push(0u8);
            key.extend(&exp_len.to_be_bytes());
        } else {
            unreachable!("RSA exponents are (much) shorter than 64KiB")
        }

        key.extend(&*self.e);
        key.extend(&*self.n);
        key.into_boxed_slice()
    }
}

impl PartialEq for RsaPublicKey {
    fn eq(&self, other: &Self) -> bool {
        /// Compare after stripping leading zeros.
        fn cmp_without_leading(a: &[u8], b: &[u8]) -> bool {
            let a = &a[a.iter().position(|&x| x != 0).unwrap_or(a.len())..];
            let b = &b[b.iter().position(|&x| x != 0).unwrap_or(b.len())..];
            if a.len() == b.len() {
                ring::constant_time::verify_slices_are_equal(a, b).is_ok()
            } else {
                false
            }
        }

        cmp_without_leading(&self.n, &other.n)
            && cmp_without_leading(&self.e, &other.e)
    }
}

#[derive(Clone, Debug)]
pub enum FromDnskeyError {
    UnsupportedAlgorithm,
    UnsupportedProtocol,
    InvalidKey,
}

#[derive(Clone, Debug)]
pub enum FromDnskeyTextError {
    Misformatted,
    FromDnskey(FromDnskeyError),
}

/// A cryptographic signature.
///
/// The format of the signature varies depending on the underlying algorithm:
///
/// - RSA: the signature is a single integer `s`, which is less than the key's
///   public modulus `n`.  `s` is encoded as bytes and ordered from most
///   significant to least significant digits.  It must be at least 64 bytes
///   long and at most 512 bytes long.  Leading zero bytes can be inserted for
///   padding.
///
///   See [RFC 3110](https://datatracker.ietf.org/doc/html/rfc3110).
///
/// - ECDSA: the signature has a fixed length (64 bytes for P-256, 96 for
///   P-384).  It is the concatenation of two fixed-length integers (`r` and
///   `s`, each of equal size).
///
///   See [RFC 6605](https://datatracker.ietf.org/doc/html/rfc6605) and [SEC 1
///   v2.0](https://www.secg.org/sec1-v2.pdf).
///
/// - EdDSA: the signature has a fixed length (64 bytes for ED25519, 114 bytes
///   for ED448).  It is the concatenation of two curve points (`R` and `S`)
///   that are encoded into bytes.
///
/// Signatures are too big to pass by value, so they are placed on the heap.
pub enum Signature {
    RsaSha1(Box<[u8]>),
    RsaSha1Nsec3Sha1(Box<[u8]>),
    RsaSha256(Box<[u8]>),
    RsaSha512(Box<[u8]>),
    EcdsaP256Sha256(Box<[u8; 64]>),
    EcdsaP384Sha384(Box<[u8; 96]>),
    Ed25519(Box<[u8; 64]>),
    Ed448(Box<[u8; 114]>),
}

//------------ Dnskey --------------------------------------------------------

/// Extensions for DNSKEY record type.
pub trait DnskeyExt {
    /// Calculates a digest from DNSKEY.
    ///
    /// See [RFC 4034, Section 5.1.4]:
    ///
    /// ```text
    /// 5.1.4.  The Digest Field
    ///   The digest is calculated by concatenating the canonical form of the
    ///   fully qualified owner name of the DNSKEY RR with the DNSKEY RDATA,
    ///   and then applying the digest algorithm.
    ///
    ///     digest = digest_algorithm( DNSKEY owner name | DNSKEY RDATA);
    ///
    ///      "|" denotes concatenation
    ///
    ///     DNSKEY RDATA = Flags | Protocol | Algorithm | Public Key.
    /// ```
    ///
    /// [RFC 4034, Section 5.1.4]: https://tools.ietf.org/html/rfc4034#section-5.1.4
    fn digest<N: ToName>(
        &self,
        name: &N,
        algorithm: DigestAlg,
    ) -> Result<digest::Digest, AlgorithmError>;
}

impl<Octets> DnskeyExt for Dnskey<Octets>
where
    Octets: AsRef<[u8]>,
{
    /// Calculates a digest from DNSKEY.
    ///
    /// See [RFC 4034, Section 5.1.4]:
    ///
    /// ```text
    /// 5.1.4.  The Digest Field
    ///   The digest is calculated by concatenating the canonical form of the
    ///   fully qualified owner name of the DNSKEY RR with the DNSKEY RDATA,
    ///   and then applying the digest algorithm.
    ///
    ///     digest = digest_algorithm( DNSKEY owner name | DNSKEY RDATA);
    ///
    ///      "|" denotes concatenation
    ///
    ///     DNSKEY RDATA = Flags | Protocol | Algorithm | Public Key.
    /// ```
    ///
    /// [RFC 4034, Section 5.1.4]: https://tools.ietf.org/html/rfc4034#section-5.1.4
    fn digest<N: ToName>(
        &self,
        name: &N,
        algorithm: DigestAlg,
    ) -> Result<digest::Digest, AlgorithmError> {
        let mut buf: Vec<u8> = Vec::new();
        with_infallible(|| {
            name.compose_canonical(&mut buf)?;
            self.compose_canonical_rdata(&mut buf)
        });

        let mut ctx = match algorithm {
            DigestAlg::SHA1 => {
                digest::Context::new(&digest::SHA1_FOR_LEGACY_USE_ONLY)
            }
            DigestAlg::SHA256 => digest::Context::new(&digest::SHA256),
            DigestAlg::SHA384 => digest::Context::new(&digest::SHA384),
            _ => {
                return Err(AlgorithmError::Unsupported);
            }
        };

        ctx.update(&buf);
        Ok(ctx.finish())
    }
}

// This needs to match the digests supported in digest.
pub fn supported_digest(d: &DigestAlg) -> bool {
    *d == DigestAlg::SHA1
        || *d == DigestAlg::SHA256
        || *d == DigestAlg::SHA384
}

//------------ Rrsig ---------------------------------------------------------

/// Extensions for DNSKEY record type.
pub trait RrsigExt {
    /// Compose the signed data according to [RC4035, Section 5.3.2](https://tools.ietf.org/html/rfc4035#section-5.3.2).
    ///
    /// ```text
    ///    Once the RRSIG RR has met the validity requirements described in
    ///    Section 5.3.1, the validator has to reconstruct the original signed
    ///    data.  The original signed data includes RRSIG RDATA (excluding the
    ///    Signature field) and the canonical form of the RRset.  Aside from
    ///    being ordered, the canonical form of the RRset might also differ from
    ///    the received RRset due to DNS name compression, decremented TTLs, or
    ///    wildcard expansion.
    /// ```
    fn signed_data<N: ToName, D, B: Composer>(
        &self,
        buf: &mut B,
        records: &mut [impl AsRef<Record<N, D>>],
    ) -> Result<(), B::AppendError>
    where
        D: RecordData + CanonicalOrd + ComposeRecordData + Sized;

    /// Return if records are expanded for a wildcard according to the
    /// information in this signature.
    fn wildcard_closest_encloser<N, D>(
        &self,
        rr: &Record<N, D>,
    ) -> Option<Name<Bytes>>
    where
        N: ToName;

    /// Attempt to use the cryptographic signature to authenticate the signed data, and thus authenticate the RRSET.
    /// The signed data is expected to be calculated as per [RFC4035, Section 5.3.2](https://tools.ietf.org/html/rfc4035#section-5.3.2).
    ///
    /// [RFC4035, Section 5.3.2](https://tools.ietf.org/html/rfc4035#section-5.3.2):
    /// ```text
    /// 5.3.3.  Checking the Signature
    ///
    ///    Once the resolver has validated the RRSIG RR as described in Section
    ///    5.3.1 and reconstructed the original signed data as described in
    ///    Section 5.3.2, the validator can attempt to use the cryptographic
    ///    signature to authenticate the signed data, and thus (finally!)
    ///    authenticate the RRset.
    ///
    ///    The Algorithm field in the RRSIG RR identifies the cryptographic
    ///    algorithm used to generate the signature.  The signature itself is
    ///    contained in the Signature field of the RRSIG RDATA, and the public
    ///    key used to verify the signature is contained in the Public Key field
    ///    of the matching DNSKEY RR(s) (found in Section 5.3.1).  [RFC4034]
    ///    provides a list of algorithm types and provides pointers to the
    ///    documents that define each algorithm's use.
    /// ```
    fn verify_signed_data(
        &self,
        dnskey: &Dnskey<impl AsRef<[u8]>>,
        signed_data: &impl AsRef<[u8]>,
    ) -> Result<(), AlgorithmError>;
}

impl<Octets: AsRef<[u8]>, TN: ToName> RrsigExt for Rrsig<Octets, TN> {
    fn signed_data<N: ToName, D, B: Composer>(
        &self,
        buf: &mut B,
        records: &mut [impl AsRef<Record<N, D>>],
    ) -> Result<(), B::AppendError>
    where
        D: RecordData + CanonicalOrd + ComposeRecordData + Sized,
    {
        // signed_data = RRSIG_RDATA | RR(1) | RR(2)...  where
        //    "|" denotes concatenation
        // RRSIG_RDATA is the wire format of the RRSIG RDATA fields
        //    with the Signature field excluded and the Signer's Name
        //    in canonical form.
        self.type_covered().compose(buf)?;
        self.algorithm().compose(buf)?;
        self.labels().compose(buf)?;
        self.original_ttl().compose(buf)?;
        self.expiration().compose(buf)?;
        self.inception().compose(buf)?;
        self.key_tag().compose(buf)?;
        self.signer_name().compose_canonical(buf)?;

        // The set of all RR(i) is sorted into canonical order.
        // See https://tools.ietf.org/html/rfc4034#section-6.3
        records.sort_by(|a, b| {
            a.as_ref().data().canonical_cmp(b.as_ref().data())
        });

        // RR(i) = name | type | class | OrigTTL | RDATA length | RDATA
        for rr in records.iter().map(|r| r.as_ref()) {
            // Handle expanded wildcards as per [RFC4035, Section 5.3.2]
            // (https://tools.ietf.org/html/rfc4035#section-5.3.2).
            let rrsig_labels = usize::from(self.labels());
            let fqdn = rr.owner();
            // Subtract the root label from count as the algorithm doesn't
            // accomodate that.
            let fqdn_labels = fqdn.iter_labels().count() - 1;
            if rrsig_labels < fqdn_labels {
                // name = "*." | the rightmost rrsig_label labels of the fqdn
                buf.append_slice(b"\x01*")?;
                match fqdn
                    .to_cow()
                    .iter_suffixes()
                    .nth(fqdn_labels - rrsig_labels)
                {
                    Some(name) => name.compose_canonical(buf)?,
                    None => fqdn.compose_canonical(buf)?,
                };
            } else {
                fqdn.compose_canonical(buf)?;
            }

            rr.rtype().compose(buf)?;
            rr.class().compose(buf)?;
            self.original_ttl().compose(buf)?;
            rr.data().compose_canonical_len_rdata(buf)?;
        }
        Ok(())
    }

    fn wildcard_closest_encloser<N, D>(
        &self,
        rr: &Record<N, D>,
    ) -> Option<Name<Bytes>>
    where
        N: ToName,
    {
        // Handle expanded wildcards as per [RFC4035, Section 5.3.2]
        // (https://tools.ietf.org/html/rfc4035#section-5.3.2).
        let rrsig_labels = usize::from(self.labels());
        let fqdn = rr.owner();
        // Subtract the root label from count as the algorithm doesn't
        // accomodate that.
        let fqdn_labels = fqdn.iter_labels().count() - 1;
        if rrsig_labels < fqdn_labels {
            // name = "*." | the rightmost rrsig_label labels of the fqdn
            Some(
                match fqdn
                    .to_cow()
                    .iter_suffixes()
                    .nth(fqdn_labels - rrsig_labels)
                {
                    Some(name) => Name::from_octets(Bytes::copy_from_slice(
                        name.as_octets(),
                    ))
                    .unwrap(),
                    None => fqdn.to_bytes(),
                },
            )
        } else {
            None
        }
    }

    fn verify_signed_data(
        &self,
        dnskey: &Dnskey<impl AsRef<[u8]>>,
        signed_data: &impl AsRef<[u8]>,
    ) -> Result<(), AlgorithmError> {
        let signature = self.signature().as_ref();
        let signed_data = signed_data.as_ref();

        // Caller needs to ensure that the signature matches the key, but enforce the algorithm match
        if self.algorithm() != dnskey.algorithm() {
            return Err(AlgorithmError::InvalidData);
        }

        // Note: Canonicalize the algorithm, otherwise matching named variants against Int(_) is not going to work
        let sec_alg = SecAlg::from_int(self.algorithm().to_int());
        match sec_alg {
            SecAlg::RSASHA1
            | SecAlg::RSASHA1_NSEC3_SHA1
            | SecAlg::RSASHA256
            | SecAlg::RSASHA512 => {
                let (algorithm, min_bytes) = match sec_alg {
                    SecAlg::RSASHA1 | SecAlg::RSASHA1_NSEC3_SHA1 => (
                        &signature::RSA_PKCS1_1024_8192_SHA1_FOR_LEGACY_USE_ONLY,
                        1024 / 8,
                    ),
                    SecAlg::RSASHA256 => (
                        &signature::RSA_PKCS1_1024_8192_SHA256_FOR_LEGACY_USE_ONLY,
                        1024 / 8,
                    ),
                    SecAlg::RSASHA512 => (
                        &signature::RSA_PKCS1_1024_8192_SHA512_FOR_LEGACY_USE_ONLY,
                        1024 / 8,
                    ),
                    _ => unreachable!(),
                };

                // The key isn't available in either PEM or DER, so use the
                // direct RSA verifier.
                let (e, n) = rsa_exponent_modulus(dnskey, min_bytes)?;
                let public_key =
                    signature::RsaPublicKeyComponents { n: &n, e: &e };
                public_key
                    .verify(algorithm, signed_data, signature)
                    .map_err(|_| AlgorithmError::BadSig)
            }
            SecAlg::ECDSAP256SHA256 | SecAlg::ECDSAP384SHA384 => {
                let algorithm = match sec_alg {
                    SecAlg::ECDSAP256SHA256 => {
                        &signature::ECDSA_P256_SHA256_FIXED
                    }
                    SecAlg::ECDSAP384SHA384 => {
                        &signature::ECDSA_P384_SHA384_FIXED
                    }
                    _ => unreachable!(),
                };

                // Add 0x4 identifier to the ECDSA pubkey as expected by ring.
                let public_key = dnskey.public_key().as_ref();
                let mut key = Vec::with_capacity(public_key.len() + 1);
                key.push(0x4);
                key.extend_from_slice(public_key);

                signature::UnparsedPublicKey::new(algorithm, &key)
                    .verify(signed_data, signature)
                    .map_err(|_| AlgorithmError::BadSig)
            }
            SecAlg::ED25519 => {
                let key = dnskey.public_key();
                signature::UnparsedPublicKey::new(&signature::ED25519, &key)
                    .verify(signed_data, signature)
                    .map_err(|_| AlgorithmError::BadSig)
            }
            _ => Err(AlgorithmError::Unsupported),
        }
    }
}

// This needs to match the algorithms supported in signed_data.
pub fn supported_algorithm(a: &SecAlg) -> bool {
    *a == SecAlg::RSASHA1
        || *a == SecAlg::RSASHA1_NSEC3_SHA1
        || *a == SecAlg::RSASHA256
        || *a == SecAlg::RSASHA512
        || *a == SecAlg::ECDSAP256SHA256
}

/// Return the RSA exponent and modulus components from DNSKEY record data.
fn rsa_exponent_modulus(
    dnskey: &Dnskey<impl AsRef<[u8]>>,
    min_len: usize,
) -> Result<(&[u8], &[u8]), AlgorithmError> {
    let public_key = dnskey.public_key().as_ref();
    if public_key.len() <= 3 {
        return Err(AlgorithmError::InvalidData);
    }

    let (pos, exp_len) = match public_key[0] {
        0 => (
            3,
            (usize::from(public_key[1]) << 8) | usize::from(public_key[2]),
        ),
        len => (1, usize::from(len)),
    };

    // Check if there's enough space for exponent and modulus.
    if public_key[pos..].len() < pos + exp_len {
        return Err(AlgorithmError::InvalidData);
    };

    // Check for minimum supported key size
    if public_key[pos..].len() < min_len {
        return Err(AlgorithmError::Unsupported);
    }

    Ok(public_key[pos..].split_at(exp_len))
}

//============ Error Types ===================================================

//------------ AlgorithmError ------------------------------------------------

/// An algorithm error during verification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgorithmError {
    Unsupported,
    BadSig,
    InvalidData,
}

//--- Display and Error

impl fmt::Display for AlgorithmError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            AlgorithmError::Unsupported => {
                f.write_str("unsupported algorithm")
            }
            AlgorithmError::BadSig => f.write_str("bad signature"),
            AlgorithmError::InvalidData => f.write_str("invalid data"),
        }
    }
}

impl error::Error for AlgorithmError {}

//============ Test ==========================================================

#[cfg(test)]
#[cfg(feature = "std")]
mod test {
    use super::*;
    use crate::base::iana::{Class, Rtype};
    use crate::base::Ttl;
    use crate::rdata::dnssec::Timestamp;
    use crate::rdata::{Mx, ZoneRecordData};
    use crate::utils::base64;
    use bytes::Bytes;
    use std::str::FromStr;

    type Name = crate::base::name::Name<Vec<u8>>;
    type Ds = crate::rdata::Ds<Vec<u8>>;
    type Dnskey = crate::rdata::Dnskey<Vec<u8>>;
    type Rrsig = crate::rdata::Rrsig<Vec<u8>, Name>;

    // Returns current root KSK/ZSK for testing (2048b)
    fn root_pubkey() -> (Dnskey, Dnskey) {
        let ksk = base64::decode::<Vec<u8>>(
            "\
            AwEAAaz/tAm8yTn4Mfeh5eyI96WSVexTBAvkMgJzkKTOiW1vkIbzxeF3+/\
            4RgWOq7HrxRixHlFlExOLAJr5emLvN7SWXgnLh4+B5xQlNVz8Og8kvArMt\
            NROxVQuCaSnIDdD5LKyWbRd2n9WGe2R8PzgCmr3EgVLrjyBxWezF0jLHwV\
            N8efS3rCj/EWgvIWgb9tarpVUDK/b58Da+sqqls3eNbuv7pr+eoZG+SrDK\
            6nWeL3c6H5Apxz7LjVc1uTIdsIXxuOLYA4/ilBmSVIzuDWfdRUfhHdY6+c\
            n8HFRm+2hM8AnXGXws9555KrUB5qihylGa8subX2Nn6UwNR1AkUTV74bU=",
        )
        .unwrap();
        let zsk = base64::decode::<Vec<u8>>(
            "\
            AwEAAeVDC34GZILwsQJy97K2Fst4P3XYZrXLyrkausYzSqEjSUulgh+iLgH\
            g0y7FIF890+sIjXsk7KLJUmCOWfYWPorNKEOKLk5Zx/4M6D3IHZE3O3m/Ea\
            hrc28qQzmTLxiMZAW65MvR2UO3LxVtYOPBEBiDgAQD47x2JLsJYtavCzNL5\
            WiUk59OgvHmDqmcC7VXYBhK8V8Tic089XJgExGeplKWUt9yyc31ra1swJX5\
            1XsOaQz17+vyLVH8AZP26KvKFiZeoRbaq6vl+hc8HQnI2ug5rA2zoz3MsSQ\
            BvP1f/HvqsWxLqwXXKyDD1QM639U+XzVB8CYigyscRP22QCnwKIU=",
        )
        .unwrap();
        (
            Dnskey::new(257, 3, SecAlg::RSASHA256, ksk).unwrap(),
            Dnskey::new(256, 3, SecAlg::RSASHA256, zsk).unwrap(),
        )
    }

    // Returns the current net KSK/ZSK for testing (1024b)
    fn net_pubkey() -> (Dnskey, Dnskey) {
        let ksk = base64::decode::<Vec<u8>>(
            "AQOYBnzqWXIEj6mlgXg4LWC0HP2n8eK8XqgHlmJ/69iuIHsa1TrHDG6TcOra/pyeGKwH0nKZhTmXSuUFGh9BCNiwVDuyyb6OBGy2Nte9Kr8NwWg4q+zhSoOf4D+gC9dEzg0yFdwT0DKEvmNPt0K4jbQDS4Yimb+uPKuF6yieWWrPYYCrv8C9KC8JMze2uT6NuWBfsl2fDUoV4l65qMww06D7n+p7RbdwWkAZ0fA63mXVXBZF6kpDtsYD7SUB9jhhfLQE/r85bvg3FaSs5Wi2BaqN06SzGWI1DHu7axthIOeHwg00zxlhTpoYCH0ldoQz+S65zWYi/fRJiyLSBb6JZOvn",
        )
        .unwrap();
        let zsk = base64::decode::<Vec<u8>>(
            "AQPW36Zs2vsDFGgdXBlg8RXSr1pSJ12NK+u9YcWfOr85we2z5A04SKQlIfyTK37dItGFcldtF7oYwPg11T3R33viKV6PyASvnuRl8QKiLk5FfGUDt1sQJv3S/9wT22Le1vnoE/6XFRyeb8kmJgz0oQB1VAO9b0l6Vm8KAVeOGJ+Qsjaq0O0aVzwPvmPtYm/i3qoAhkaMBUpg6RrF5NKhRyG3",
        )
        .unwrap();
        (
            Dnskey::new(257, 3, SecAlg::RSASHA256, ksk).unwrap(),
            Dnskey::new(256, 3, SecAlg::RSASHA256, zsk).unwrap(),
        )
    }

    #[test]
    fn dnskey_digest() {
        let (dnskey, _) = root_pubkey();
        let owner = Name::root();
        let expected = Ds::new(
            20326,
            SecAlg::RSASHA256,
            DigestAlg::SHA256,
            base64::decode::<Vec<u8>>(
                "4G1EuAuPHTmpXAsNfGXQhFjogECbvGg0VxBCN8f47I0=",
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            dnskey.digest(&owner, DigestAlg::SHA256).unwrap().as_ref(),
            expected.digest()
        );
    }

    #[test]
    fn dnskey_digest_unsupported() {
        let (dnskey, _) = root_pubkey();
        let owner = Name::root();
        assert!(dnskey.digest(&owner, DigestAlg::GOST).is_err());
    }

    fn rrsig_verify_dnskey(ksk: Dnskey, zsk: Dnskey, rrsig: Rrsig) {
        let mut records: Vec<_> = [&ksk, &zsk]
            .iter()
            .cloned()
            .map(|x| {
                Record::new(
                    rrsig.signer_name().clone(),
                    Class::IN,
                    Ttl::from_secs(0),
                    x.clone(),
                )
            })
            .collect();
        let signed_data = {
            let mut buf = Vec::new();
            rrsig.signed_data(&mut buf, records.as_mut_slice()).unwrap();
            Bytes::from(buf)
        };

        // Test that the KSK is sorted after ZSK key
        assert_eq!(ksk.key_tag(), rrsig.key_tag());
        assert_eq!(ksk.key_tag(), records[1].data().key_tag());

        // Test verifier
        assert!(rrsig.verify_signed_data(&ksk, &signed_data).is_ok());
        assert!(rrsig.verify_signed_data(&zsk, &signed_data).is_err());
    }

    #[test]
    fn rrsig_verify_rsa_sha256() {
        // Test 2048b long key
        let (ksk, zsk) = root_pubkey();
        let rrsig = Rrsig::new(
            Rtype::DNSKEY,
            SecAlg::RSASHA256,
            0,
            Ttl::from_secs(172800),
            1560211200.into(),
            1558396800.into(),
            20326,
            Name::root(),
            base64::decode::<Vec<u8>>(
                "otBkINZAQu7AvPKjr/xWIEE7+SoZtKgF8bzVynX6bfJMJuPay8jPvNmwXkZOdSoYlvFp0bk9JWJKCh8y5uoNfMFkN6OSrDkr3t0E+c8c0Mnmwkk5CETH3Gqxthi0yyRX5T4VlHU06/Ks4zI+XAgl3FBpOc554ivdzez8YCjAIGx7XgzzooEb7heMSlLc7S7/HNjw51TPRs4RxrAVcezieKCzPPpeWBhjE6R3oiSwrl0SBD4/yplrDlr7UHs/Atcm3MSgemdyr2sOoOUkVQCVpcj3SQQezoD2tCM7861CXEQdg5fjeHDtz285xHt5HJpA5cOcctRo4ihybfow/+V7AQ==",
            )
            .unwrap()
        ).unwrap();
        rrsig_verify_dnskey(ksk, zsk, rrsig);

        // Test 1024b long key
        let (ksk, zsk) = net_pubkey();
        let rrsig = Rrsig::new(
            Rtype::DNSKEY,
            SecAlg::RSASHA256,
            1,
            Ttl::from_secs(86400),
            Timestamp::from_str("20210921162830").unwrap(),
            Timestamp::from_str("20210906162330").unwrap(),
            35886,
            "net.".parse::<Name>().unwrap(),
            base64::decode::<Vec<u8>>(
                "j1s1IPMoZd0mbmelNVvcbYNe2tFCdLsLpNCnQ8xW6d91ujwPZ2yDlc3lU3hb+Jq3sPoj+5lVgB7fZzXQUQTPFWLF7zvW49da8pWuqzxFtg6EjXRBIWH5rpEhOcr+y3QolJcPOTx+/utCqt2tBKUUy3LfM6WgvopdSGaryWdwFJPW7qKHjyyLYxIGx5AEuLfzsA5XZf8CmpUheSRH99GRZoIB+sQzHuelWGMQ5A42DPvOVZFmTpIwiT2QaIpid4nJ7jNfahfwFrCoS+hvqjK9vktc5/6E/Mt7DwCQDaPt5cqDfYltUitQy+YA5YP5sOhINChYadZe+2N80OA+RKz0mA==",
            )
            .unwrap()
        ).unwrap();
        rrsig_verify_dnskey(ksk, zsk, rrsig.clone());

        // Test that 512b short RSA DNSKEY is not supported (too short)
        let data = base64::decode::<Vec<u8>>(
            "AwEAAcFcGsaxxdgiuuGmCkVImy4h99CqT7jwY3pexPGcnUFtR2Fh36BponcwtkZ4cAgtvd4Qs8PkxUdp6p/DlUmObdk=",
        )
        .unwrap();

        let short_key = Dnskey::new(256, 3, SecAlg::RSASHA256, data).unwrap();
        let err = rrsig
            .verify_signed_data(&short_key, &vec![0; 100])
            .unwrap_err();
        assert_eq!(err, AlgorithmError::Unsupported);
    }

    #[test]
    fn rrsig_verify_ecdsap256_sha256() {
        let (ksk, zsk) = (
            Dnskey::new(
                257,
                3,
                SecAlg::ECDSAP256SHA256,
                base64::decode::<Vec<u8>>(
                    "mdsswUyr3DPW132mOi8V9xESWE8jTo0dxCjjnopKl+GqJxpVXckHAe\
                    F+KkxLbxILfDLUT0rAK9iUzy1L53eKGQ==",
                )
                .unwrap(),
            )
            .unwrap(),
            Dnskey::new(
                256,
                3,
                SecAlg::ECDSAP256SHA256,
                base64::decode::<Vec<u8>>(
                    "oJMRESz5E4gYzS/q6XDrvU1qMPYIjCWzJaOau8XNEZeqCYKD5ar0IR\
                    d8KqXXFJkqmVfRvMGPmM1x8fGAa2XhSA==",
                )
                .unwrap(),
            )
            .unwrap(),
        );

        let owner = Name::from_str("cloudflare.com.").unwrap();
        let rrsig = Rrsig::new(
            Rtype::DNSKEY,
            SecAlg::ECDSAP256SHA256,
            2,
            Ttl::from_secs(3600),
            1560314494.into(),
            1555130494.into(),
            2371,
            owner,
            base64::decode::<Vec<u8>>(
                "8jnAGhG7O52wmL065je10XQztRX1vK8P8KBSyo71Z6h5wAT9+GFxKBaE\
                zcJBLvRmofYFDAhju21p1uTfLaYHrg==",
            )
            .unwrap(),
        )
        .unwrap();
        rrsig_verify_dnskey(ksk, zsk, rrsig);
    }

    #[test]
    fn rrsig_verify_ed25519() {
        let (ksk, zsk) = (
            Dnskey::new(
                257,
                3,
                SecAlg::ED25519,
                base64::decode::<Vec<u8>>(
                    "m1NELLVVQKl4fHVn/KKdeNO0PrYKGT3IGbYseT8XcKo=",
                )
                .unwrap(),
            )
            .unwrap(),
            Dnskey::new(
                256,
                3,
                SecAlg::ED25519,
                base64::decode::<Vec<u8>>(
                    "2tstZAjgmlDTePn0NVXrAHBJmg84LoaFVxzLl1anjGI=",
                )
                .unwrap(),
            )
            .unwrap(),
        );

        let owner =
            Name::from_octets(Vec::from(b"\x07ED25519\x02nl\x00".as_ref()))
                .unwrap();
        let rrsig = Rrsig::new(
            Rtype::DNSKEY,
            SecAlg::ED25519,
            2,
            Ttl::from_secs(3600),
            1559174400.into(),
            1557360000.into(),
            45515,
            owner,
            base64::decode::<Vec<u8>>(
                "hvPSS3E9Mx7lMARqtv6IGiw0NE0uz0mZewndJCHTkhwSYqlasUq7KfO5\
                QdtgPXja7YkTaqzrYUbYk01J8ICsAA==",
            )
            .unwrap(),
        )
        .unwrap();
        rrsig_verify_dnskey(ksk, zsk, rrsig);
    }

    #[test]
    fn rrsig_verify_generic_type() {
        let (ksk, zsk) = root_pubkey();
        let rrsig = Rrsig::new(
            Rtype::DNSKEY,
            SecAlg::RSASHA256,
            0,
            Ttl::from_secs(172800),
            1560211200.into(),
            1558396800.into(),
            20326,
            Name::root(),
            base64::decode::<Vec<u8>>(
                "otBkINZAQu7AvPKjr/xWIEE7+SoZtKgF8bzVynX6bfJMJuPay8jPvNmwXkZ\
                OdSoYlvFp0bk9JWJKCh8y5uoNfMFkN6OSrDkr3t0E+c8c0Mnmwkk5CETH3Gq\
                xthi0yyRX5T4VlHU06/Ks4zI+XAgl3FBpOc554ivdzez8YCjAIGx7XgzzooE\
                b7heMSlLc7S7/HNjw51TPRs4RxrAVcezieKCzPPpeWBhjE6R3oiSwrl0SBD4\
                /yplrDlr7UHs/Atcm3MSgemdyr2sOoOUkVQCVpcj3SQQezoD2tCM7861CXEQ\
                dg5fjeHDtz285xHt5HJpA5cOcctRo4ihybfow/+V7AQ==",
            )
            .unwrap(),
        )
        .unwrap();

        let mut records: Vec<Record<Name, ZoneRecordData<Vec<u8>, Name>>> =
            [&ksk, &zsk]
                .iter()
                .cloned()
                .map(|x| {
                    let data = ZoneRecordData::from(x.clone());
                    Record::new(
                        rrsig.signer_name().clone(),
                        Class::IN,
                        Ttl::from_secs(0),
                        data,
                    )
                })
                .collect();

        let signed_data = {
            let mut buf = Vec::new();
            rrsig.signed_data(&mut buf, records.as_mut_slice()).unwrap();
            Bytes::from(buf)
        };

        assert!(rrsig.verify_signed_data(&ksk, &signed_data).is_ok());
    }

    #[test]
    fn rrsig_verify_wildcard() {
        let key = Dnskey::new(
            256,
            3,
            SecAlg::RSASHA1,
            base64::decode::<Vec<u8>>(
                "AQOy1bZVvpPqhg4j7EJoM9rI3ZmyEx2OzDBVrZy/lvI5CQePxX\
                HZS4i8dANH4DX3tbHol61ek8EFMcsGXxKciJFHyhl94C+NwILQd\
                zsUlSFovBZsyl/NX6yEbtw/xN9ZNcrbYvgjjZ/UVPZIySFNsgEY\
                vh0z2542lzMKR4Dh8uZffQ==",
            )
            .unwrap(),
        )
        .unwrap();
        let rrsig = Rrsig::new(
            Rtype::MX,
            SecAlg::RSASHA1,
            2,
            Ttl::from_secs(3600),
            Timestamp::from_str("20040509183619").unwrap(),
            Timestamp::from_str("20040409183619").unwrap(),
            38519,
            Name::from_str("example.").unwrap(),
            base64::decode::<Vec<u8>>(
                "OMK8rAZlepfzLWW75Dxd63jy2wswESzxDKG2f9AMN1CytCd10cYI\
                 SAxfAdvXSZ7xujKAtPbctvOQ2ofO7AZJ+d01EeeQTVBPq4/6KCWhq\
                 e2XTjnkVLNvvhnc0u28aoSsG0+4InvkkOHknKxw4kX18MMR34i8lC\
                 36SR5xBni8vHI=",
            )
            .unwrap(),
        )
        .unwrap();
        let record = Record::new(
            Name::from_str("a.z.w.example.").unwrap(),
            Class::IN,
            Ttl::from_secs(3600),
            Mx::new(1, Name::from_str("ai.example.").unwrap()),
        );
        let signed_data = {
            let mut buf = Vec::new();
            rrsig.signed_data(&mut buf, &mut [record]).unwrap();
            Bytes::from(buf)
        };

        // Test that the key matches RRSIG
        assert_eq!(key.key_tag(), rrsig.key_tag());

        // Test verifier
        assert_eq!(rrsig.verify_signed_data(&key, &signed_data), Ok(()));
    }
}
