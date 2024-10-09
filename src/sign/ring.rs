//! DNSSEC signing using `ring`.

#![cfg(feature = "ring")]
#![cfg_attr(docsrs, doc(cfg(feature = "ring")))]

use std::vec::Vec;

use crate::base::iana::SecAlg;

use super::generic;

/// A key pair backed by `ring`.
pub enum SecretKey<'a> {
    /// An RSA/SHA-256 keypair.
    RsaSha256 {
        key: ring::signature::RsaKeyPair,
        rng: &'a dyn ring::rand::SecureRandom,
    },

    /// An Ed25519 keypair.
    Ed25519(ring::signature::Ed25519KeyPair),
}

impl<'a> SecretKey<'a> {
    /// Use a generic keypair with `ring`.
    pub fn import<B: AsRef<[u8]> + AsMut<[u8]>>(
        key: generic::SecretKey<B>,
        rng: &'a dyn ring::rand::SecureRandom,
    ) -> Result<Self, ImportError> {
        match &key {
            generic::SecretKey::RsaSha256(k) => {
                let components = ring::rsa::KeyPairComponents {
                    public_key: ring::rsa::PublicKeyComponents {
                        n: k.n.as_ref(),
                        e: k.e.as_ref(),
                    },
                    d: k.d.as_ref(),
                    p: k.p.as_ref(),
                    q: k.q.as_ref(),
                    dP: k.d_p.as_ref(),
                    dQ: k.d_q.as_ref(),
                    qInv: k.q_i.as_ref(),
                };
                ring::signature::RsaKeyPair::from_components(&components)
                    .map_err(|_| ImportError::InvalidKey)
                    .map(|key| Self::RsaSha256 { key, rng })
            }
            // TODO: Support ECDSA.
            generic::SecretKey::Ed25519(k) => {
                let k = k.as_ref();
                ring::signature::Ed25519KeyPair::from_seed_unchecked(k)
                    .map_err(|_| ImportError::InvalidKey)
                    .map(Self::Ed25519)
            }
            _ => Err(ImportError::UnsupportedAlgorithm),
        }
    }

    /// Export this key into a generic public key.
    pub fn export_public<B>(&self) -> generic::PublicKey<B>
    where
        B: AsRef<[u8]> + From<Vec<u8>>,
    {
        match self {
            Self::RsaSha256 { key, rng: _ } => {
                let components: ring::rsa::PublicKeyComponents<Vec<u8>> =
                    key.public().into();
                generic::PublicKey::RsaSha256(generic::RsaPublicKey {
                    n: components.n.into(),
                    e: components.e.into(),
                })
            }
            Self::Ed25519(key) => {
                use ring::signature::KeyPair;
                let key = key.public_key().as_ref();
                generic::PublicKey::Ed25519(key.try_into().unwrap())
            }
        }
    }
}

/// An error in importing a key into `ring`.
pub enum ImportError {
    /// The requested algorithm was not supported.
    UnsupportedAlgorithm,

    /// The provided keypair was invalid.
    InvalidKey,
}

impl<'a> super::Sign<Vec<u8>> for SecretKey<'a> {
    type Error = ring::error::Unspecified;

    fn algorithm(&self) -> SecAlg {
        match self {
            Self::RsaSha256 { .. } => SecAlg::RSASHA256,
            Self::Ed25519(_) => SecAlg::ED25519,
        }
    }

    fn sign(&self, data: &[u8]) -> Result<Vec<u8>, Self::Error> {
        match self {
            Self::RsaSha256 { key, rng } => {
                let mut buf = vec![0u8; key.public().modulus_len()];
                let pad = &ring::signature::RSA_PKCS1_SHA256;
                key.sign(pad, *rng, data, &mut buf)?;
                Ok(buf)
            }
            Self::Ed25519(key) => Ok(key.sign(data).as_ref().to_vec()),
        }
    }
}
