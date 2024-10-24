//! DNSSEC signing using built-in backends.

use core::fmt;
use std::sync::Arc;

use ::ring::rand::SystemRandom;

use crate::{
    base::iana::SecAlg,
    validate::{RawPublicKey, Signature},
};

use super::{
    generic::{self, GenerateParams},
    openssl, ring, SignError, SignRaw,
};

//----------- SecretKey ------------------------------------------------------

/// A key pair based on a built-in backend.
///
/// This supports any built-in backend (currently, that is OpenSSL and Ring).
/// Wherever possible, the Ring backend is preferred over OpenSSL -- but for
/// more uncommon or insecure algorithms, that Ring does not support, OpenSSL
/// must be used.
pub enum SecretKey {
    /// A key backed by Ring.
    #[cfg(feature = "ring")]
    Ring(ring::SecretKey),

    /// A key backed by OpenSSL.
    OpenSSL(openssl::SecretKey),
}

//--- Conversion to and from generic keys

impl SecretKey {
    /// Use a generic secret key with OpenSSL.
    pub fn from_generic(
        secret: &generic::SecretKey,
        public: &RawPublicKey,
    ) -> Result<Self, FromGenericError> {
        // Prefer Ring if it is available.
        #[cfg(feature = "ring")]
        match public {
            RawPublicKey::RsaSha1(k)
            | RawPublicKey::RsaSha1Nsec3Sha1(k)
            | RawPublicKey::RsaSha256(k)
            | RawPublicKey::RsaSha512(k)
                if k.n.len() >= 2048 / 8 =>
            {
                let rng = Arc::new(SystemRandom::new());
                let key = ring::SecretKey::from_generic(secret, public, rng)?;
                return Ok(Self::Ring(key));
            }

            RawPublicKey::EcdsaP256Sha256(_)
            | RawPublicKey::EcdsaP384Sha384(_) => {
                let rng = Arc::new(SystemRandom::new());
                let key = ring::SecretKey::from_generic(secret, public, rng)?;
                return Ok(Self::Ring(key));
            }

            RawPublicKey::Ed25519(_) => {
                let rng = Arc::new(SystemRandom::new());
                let key = ring::SecretKey::from_generic(secret, public, rng)?;
                return Ok(Self::Ring(key));
            }

            _ => {}
        }

        // Fall back to OpenSSL.
        Ok(Self::OpenSSL(openssl::SecretKey::from_generic(
            secret, public,
        )?))
    }
}

//--- SignRaw

impl SignRaw for SecretKey {
    fn algorithm(&self) -> SecAlg {
        match self {
            #[cfg(feature = "ring")]
            Self::Ring(key) => key.algorithm(),
            Self::OpenSSL(key) => key.algorithm(),
        }
    }

    fn raw_public_key(&self) -> RawPublicKey {
        match self {
            #[cfg(feature = "ring")]
            Self::Ring(key) => key.raw_public_key(),
            Self::OpenSSL(key) => key.raw_public_key(),
        }
    }

    fn sign_raw(&self, data: &[u8]) -> Result<Signature, SignError> {
        match self {
            #[cfg(feature = "ring")]
            Self::Ring(key) => key.sign_raw(data),
            Self::OpenSSL(key) => key.sign_raw(data),
        }
    }
}

//----------- generate() -----------------------------------------------------

/// Generate a new secret key for the given algorithm.
pub fn generate(params: GenerateParams) -> Result<SecretKey, GenerateError> {
    // TODO: Support key generation in Ring.
    Ok(SecretKey::OpenSSL(openssl::generate(params)?))
}

//============ Error Types ===================================================

//----------- FromGenericError -----------------------------------------------

/// An error in importing a key.
#[derive(Clone, Debug)]
pub enum FromGenericError {
    /// The requested algorithm was not supported.
    UnsupportedAlgorithm,

    /// The key's parameters were invalid.
    InvalidKey,

    /// The implementation does not allow such weak keys.
    WeakKey,

    /// An implementation failure occurred.
    ///
    /// This includes memory allocation failures.
    Implementation,
}

//--- Conversions

impl From<ring::FromGenericError> for FromGenericError {
    fn from(value: ring::FromGenericError) -> Self {
        match value {
            ring::FromGenericError::UnsupportedAlgorithm => {
                Self::UnsupportedAlgorithm
            }
            ring::FromGenericError::InvalidKey => Self::InvalidKey,
            ring::FromGenericError::WeakKey => Self::WeakKey,
        }
    }
}

impl From<openssl::FromGenericError> for FromGenericError {
    fn from(value: openssl::FromGenericError) -> Self {
        match value {
            openssl::FromGenericError::UnsupportedAlgorithm => {
                Self::UnsupportedAlgorithm
            }
            openssl::FromGenericError::InvalidKey => Self::InvalidKey,
            openssl::FromGenericError::Implementation => Self::Implementation,
        }
    }
}

//--- Formatting

impl fmt::Display for FromGenericError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnsupportedAlgorithm => "algorithm not supported",
            Self::InvalidKey => "malformed or insecure private key",
            Self::WeakKey => "key too weak to be supported",
            Self::Implementation => "an internal error occurred",
        })
    }
}

//--- Error

impl std::error::Error for FromGenericError {}

//----------- GenerateError --------------------------------------------------

/// An error in generating a key.
#[derive(Clone, Debug)]
pub enum GenerateError {
    /// The requested algorithm was not supported.
    UnsupportedAlgorithm,

    /// An implementation failure occurred.
    ///
    /// This includes memory allocation failures.
    Implementation,
}

//--- Conversion

impl From<openssl::GenerateError> for GenerateError {
    fn from(value: openssl::GenerateError) -> Self {
        match value {
            openssl::GenerateError::UnsupportedAlgorithm => {
                Self::UnsupportedAlgorithm
            }
            openssl::GenerateError::Implementation => Self::Implementation,
        }
    }
}

//--- Formatting

impl fmt::Display for GenerateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnsupportedAlgorithm => "algorithm not supported",
            Self::Implementation => "an internal error occurred",
        })
    }
}

//--- Error

impl std::error::Error for GenerateError {}
