//! Resource data implementations.
//!
//! This module will eventually contain implementations for the record data
//! for all defined resource record types.
//!
//! The types are named identically to the [`Rtype`] variant they implement.
//! They are grouped into submodules for the RFCs they are defined in. All
//! types are also re-exported at the top level here. Ie., for the AAAA
//! record type, you can simple `use domain::rdata::Aaaa` instead of
//! `use domain::rdata::rfc3596::Aaaa` which nobody could possibly remember.
//! There are, however, some helper data types defined here and there which
//! are not re-exported to keep things somewhat tidy.
//!
//! See the [`Rtype`] enum for the complete set of record types and,
//! consequently, those types that are still missing.
//!
//! [`Rtype`]: ../iana/enum.Rtype.html

pub mod rfc1035;
pub mod rfc2782;
pub mod rfc2845;
pub mod rfc3596;
pub mod rfc4034;
pub mod rfc5155;
pub mod rfc7344;

#[macro_use]
mod macros;

// The rdata_types! macro (defined in self::macros) reexports the record data
// types here and creates the MasterRecordData and AllRecordData enums
// containing all record types that can appear in master files or all record
// types that exist.
//
// All record data types listed here should have the same name as the
// `Rtype` variant they implement.
//
// Add any new module here and then add all record types in that module that
// can appear in master files under "master" and all others under "pseudo".
// In both cases, if your type is generic over a domain name type, add `<N>`
// to it (it can’t be over anything else, so if you have more type arguments,
// you might have to either newtype with those removes or, God forbid, modify
// the macro). Each type entry has to be followed by a comma, even the last
// one.
rdata_types! {
    rfc1035::{
        master {
            A,
            Cname<N>,
            Hinfo,
            Mb<N>,
            Md<N>,
            Mf<N>,
            Minfo<N>,
            Mr<N>,
            Mx<N>,
            Ns<N>,
            Ptr<N>,
            Soa<N>,
            Txt,
            Wks,
        }
        pseudo {
            Null,
        }
    }
    rfc2782::{
        master {
            Srv<N>,
        }
    }
    rfc2845::{
        pseudo {
            Tsig<N>,
        }
    }
    rfc3596::{
        master {
            Aaaa,
        }
    }
    rfc4034::{
        master {
            Dnskey,
            Rrsig,
            Nsec<N>,
            Ds,
        }
    }
    rfc5155::{
        master {
            Nsec3,
            Nsec3param,
        }
    }
    rfc7344::{
        master {
            Cdnskey,
            Cds,
        }
    }
}

pub mod parsed {
    pub use super::rfc1035::parsed::*;
    pub use super::rfc2782::parsed::*;
    pub use super::rfc3596::parsed::*;
    pub use super::rfc4034::parsed::*;
    pub use super::rfc5155::parsed::*;
    pub use super::rfc7344::parsed::*;
}

use std::{error, fmt};
use bytes::{BufMut, Bytes, BytesMut};
use crate::compose::{Compose, Compress, Compressor};
use crate::iana::Rtype;
use crate::master::scan::{CharSource, Scan, Scanner, ScanError, SyntaxError};
use crate::opt::Opt;
use crate::parse::{ParseAll, Parser, ShortBuf};

//----------- RecordData -----------------------------------------------------

/// A type that represents record data.
///
/// The type needs to be able to encode the record data into a DNS message
/// via the [`Compose`] and [`Compress`] traits. In addition, it needs to be
/// able to provide the record type of a record with a value’s data via the
/// [`rtype`] method.
///
/// [`Compose`]: ../compose/trait.Compose.html
/// [`Compress`]: ../compose/trait.Compress.html
/// [`rtype`]: #method.rtype
pub trait RecordData: Compose + Compress + Sized {
    /// Returns the record type associated with this record data instance.
    ///
    /// This is a method rather than an associated function to allow one
    /// type to be used for several real record types.
    fn rtype(&self) -> Rtype;
}


//------------ ParseRecordData -----------------------------------------------

/// A record data type that can be parsed from a message.
///
/// When record data types are generic – typically over a domain name type –,
/// they may not in all cases be parseable. They may still represent record
/// data to be used when constructing the message.
///
/// To reflect this asymmetry, parsing of record data has its own trait.
pub trait ParseRecordData: RecordData {
    /// The type of an error returned when parsing fails.
    type Err: error::Error;

    /// Parses the record data.
    ///
    /// The record data is for a record of type `rtype`. The function may
    /// decide whether it wants to parse data for that type. It should return
    /// `Ok(None)` if it doesn’t. The data is `rdlen` bytes long and starts
    /// at the current position of `parser`. There is no guarantee that the
    /// parser will have `rdlen` bytes left. If it doesn’t, the function
    /// should produce an error.
    ///
    /// If the function doesn’t want to process the data, it must not touch
    /// the parser. In particual, it must not advance it.
    fn parse_data(rtype: Rtype, parser: &mut Parser, rdlen: usize)
                  -> Result<Option<Self>, Self::Err>;
}


//------------ RtypeRecordData -----------------------------------------------

/// A type for record data for a single specific record type.
///
/// If a record data type only ever processes one single record type, things
/// can be a lot simpler. The type can be given as an associated constant
/// which can be used to implement [`RecordData`]. In addition, parsing can
/// be done atop an implementation of the [`ParseAll`] trait.
///
/// This trait provides such a simplification by providing [`RecordData`]
/// for all types implementing it and the other requirements for
/// [`RecordData`]. If the type additionally implements [`ParseAll`], it will
/// also receive a [`ParseRecordData`] implementation.
///
/// [`RecordData`]: trait.RecordData.html
/// [`ParseRecordData`]: trait.ParseRecordData.html
/// [`ParseAll`]: ../parse/trait.ParseAll.html
pub trait RtypeRecordData {
    /// The record type of a value of this type.
    const RTYPE: Rtype;
}

impl<T: RtypeRecordData + Compose + Compress + Sized> RecordData for T {
    fn rtype(&self) -> Rtype { Self::RTYPE }
}

impl<T: RtypeRecordData + ParseAll + Compose + Compress + Sized>
            ParseRecordData for T {
    type Err = <Self as ParseAll>::Err;

    fn parse_data(rtype: Rtype, parser: &mut Parser, rdlen: usize)
                  -> Result<Option<Self>, Self::Err> {
        if rtype == Self::RTYPE {
            Self::parse_all(parser, rdlen).map(Some)
        }
        else {
            Ok(None)
        }
    }
}


//------------ UnknownRecordData ---------------------------------------------

/// A type for parsing any type of record data.
///
/// This type accepts any record type and stores a reference to the plain
/// binary record data in the message.
///
/// Because some record types allow compressed domain names in their record
/// data yet values only contain the data’s own bytes, this type cannot be
/// used safely with these record types.
///
/// [RFC 3597] limits the types for which compressed names are allowed in the
/// record data to those efined in [RFC 1035] itself. Specific types for all
/// these record types exist in [`domain::rdata::rfc1035`].
///
/// Ultimately, you should only use this type for record types for which there
/// is no implementation available in this crate.
///
/// [RFC 1035]: https://tools.ietf.org/html/rfc1035
/// [RFC 3597]: https://tools.ietf.org/html/rfc3597
/// [`domain::rdata::rfc1035]: ../../rdata/rfc1035/index.html
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct UnknownRecordData {
    /// The record type of this data.
    rtype: Rtype,

    /// The record data.
    data: Bytes,
}

impl UnknownRecordData {
    /// Creates generic record data from a bytes value contain the data.
    pub fn from_bytes(rtype: Rtype, data: Bytes) -> Self {
        UnknownRecordData { rtype, data }
    }

    /// Returns the record type this data is for.
    pub fn rtype(&self) -> Rtype {
        self.rtype
    }

    /// Returns a reference to the record data.
    pub fn data(&self) -> &Bytes {
        &self.data
    }

    /// Scans the record data.
    ///
    /// This isn’t implemented via `Scan`, because we need the record type.
    pub fn scan<C: CharSource>(rtype: Rtype, scanner: &mut Scanner<C>)
                               -> Result<Self, ScanError> {
        scanner.skip_literal("\\#")?;
        let mut len = u16::scan(scanner)? as usize;
        let mut res = BytesMut::with_capacity(len);
        while len > 0 {
            len = scanner.scan_word(
                (&mut res, len, None), // buffer and optional first char
                |&mut (ref mut res, ref mut len, ref mut first), symbol| {
                    if *len == 0 {
                        return Err(SyntaxError::LongGenericData)
                    }
                    let ch = symbol.into_digit(16)? as u8;
                    if let Some(ch1) = *first {
                        res.put_u8(ch1 << 4 | ch);
                        *len -= 1;
                    }
                    else {
                        *first = Some(ch)
                    }
                    Ok(())
                },
                |(_, len, first)| {
                    if first.is_some() {
                        Err(SyntaxError::UnevenHexString)
                    }
                    else {
                        Ok(len)
                    }
                }
            )?
        }
        Ok(UnknownRecordData::from_bytes(rtype, res.freeze()))
    }
}


//--- Compose, and Compress

impl Compose for UnknownRecordData {
    fn compose_len(&self) -> usize {
        self.data.len()
    }

    fn compose<B: BufMut>(&self, buf: &mut B) {
        buf.put_slice(self.data.as_ref())
    }
}

impl Compress for UnknownRecordData {
    fn compress(&self, buf: &mut Compressor) -> Result<(), ShortBuf> {
        buf.compose(self)
    }
}


//--- RecordData and ParseRecordData

impl RecordData for UnknownRecordData {
    fn rtype(&self) -> Rtype {
        self.rtype
    }
}

impl ParseRecordData for UnknownRecordData {
    type Err = ShortBuf;

    fn parse_data(rtype: Rtype, parser: &mut Parser, rdlen: usize)
                  -> Result<Option<Self>, Self::Err> {
        parser.parse_bytes(rdlen)
              .map(|data| Some(Self::from_bytes(rtype, data)))
    }
}


//--- Display

impl fmt::Display for UnknownRecordData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\\# {}", self.data.len())?;
        for ch in self.data.as_ref() {
            write!(f, " {:02x}", *ch)?
        }
        Ok(())
    }
}

