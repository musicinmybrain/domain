//! A vectorized zonefile parser.
//!
//! This is a performance-oriented parser for large zonefiles, based on special
//! SIMD (Single-Instruction, Multiple-Data) instruction sets found on modern
//! CPUs.  It is faster but less flexible than using the [`Scan`] trait system.
//!
//! [`Scan`]: crate::base::scan::Scan
//!
//! # Algorithm
//!
//! A _lexer_ extracts all the records in a file and sorts them by record type.
//! Then, a specialized parser for each record type is executed over its set of
//! records.  This two-stage approach can leverage vectorized instructions more
//! often, since each stage has less work to do.
//!
//! The lexer divides the input file into _segments_ and processes them in
//! parallel.  A combining pass merges the record data from each segment and
//! organizes the records by their type.  Segmentation occurs based on the file
//! size, not its contents, so segment boundaries might lie in the middle of a
//! record or token; accounting for this correctly is complicated.
//!
//! Within a segment, processing occurs serially.  New records are looked for at
//! the start of each line.  The goal is to identify the lexical components of
//! every record -- the domain name, the time-to-live (TTL), the record class
//! (which we assume is always `IN`), the record type, and the record data.
//!
//! The algorithm optimistically assumes that the input zonefile is formatted
//! correctly.  It can detect a format error but it cannot provide useful error
//! messages or even pinpoint the location of the error.  If an error occurs, a
//! slower but smarter parser must be used to collect human-readable errors.
//!
//! # Lexing a Record
//!
//! A record consists of five components.  They are lexed thusly:
//!
//! - The domain name is always located at the start of the line, with no
//!   intervening whitespace.  It is identified by the regex `\n[A-Za-z]`.
//!
//!   An implicit domain name is identified by the regex `\n[ \t]`.
//!
//! - A TTL token is separated by whitespace and starts with a digit.  It is
//!   identified by the regex `[ \t][0-9]`.
//!
//! - The record class token is almost always `IN` (Internet).  While we could
//!   support other kinds of tokens, it would be hard to distinguish them from
//!   record data tokens, so we simply don't.  It is identified by the regex
//!   `[ \t]IN[ \t]`.
//!
//! - The record type token is the first token matching `[ \t][A-Za-z]` on each
//!   line that is also not a record class.
//!
//! - The record data is the remaining content of the line.
//!
//! # Architecture Support
//!
//! At the moment, this implementation targets x86-64 CPUs which support AVX2.
//! More architectures and SIMD extensions will be supported in the future.
