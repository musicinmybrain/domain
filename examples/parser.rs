use core::arch::x86_64::*;
use std::io::{self, Read};
use std::{env, fs::File};

#[derive(Debug)]
struct Frame<'a> {
    /// The input data for a frame.
    ///
    /// There must be at least 8 bytes of additional memory lying beyond the end
    /// of this array that are safe to read.
    input: &'a [u8; 65536],

    /// Entries parsed from this frame.
    ///
    /// A frame can contain at most 8K entries, since entries with fewer than
    /// 8 bytes are virtually impossible.  Nicely, this means another 64KiB of
    /// data storage.
    entries: &'a mut [Entry; 8192],
}

impl<'a> Frame<'a> {
    /// Lex the frame into entries.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn lex(self) -> Result<(), Error> {
        // The start and limit of the segment.
        let (start, limit) = self.segment()?;

        // The current offset in the segment.
        let mut offset = start;

        // Whether the segment is still working.
        let mut working = _mm256_set1_epi64x(!0);

        let mut count = 0usize;

        loop {
            // Load the next 8 bytes of the segment.
            let view = self.view(offset, working);

            // Move to the next view.
            offset = self.next_offset(view, offset, working);

            // Stop processing if the segment has ended.
            working = _mm256_cmpgt_epi64(limit, offset);

            count += 1;

            // If all processing is finished, stop.
            if _mm256_testz_si256(working, working) != 0 {
                break;
            }
        }

        println!("Performed {count} iterations");

        Ok(())
    }

    /// The offset of the next view.
    ///
    /// The next view will begin at a new word (if one was found within the
    /// current view), or at the next 8 bytes of the current word.
    ///
    /// Latency (view -> return): 17c
    /// Latency (offset -> return): 1c
    /// Latency (working -> return): 2c
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn next_offset(
        &self,
        view: __m256i,
        offset: __m256i,
        working: __m256i,
    ) -> __m256i {
        // Mark bytes in the current word.
        let w = self.word_mask(view);

        // Increment the number of set bytes, saturating to 8.
        let w = _mm256_or_si256(w, _mm256_set1_epi64x((0xFFu64 << 56) as _));

        // Count the number of set bytes.
        let c = Self::count_bytes(w);

        // Zero the count if the segment is finished working.
        let c = _mm256_and_si256(c, working);

        // Add the count to the current offset.
        _mm256_add_epi64(offset, c)
    }

    /// Whether this is the last token on the line.
    ///
    /// This is all-ones if the whitespace terminating the current token is a
    /// newline (so that the next token will follow it).
    ///
    /// Latency (view -> return): 7c
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn end_of_line(&self, view: __m256i) -> __m256i {
        // If the mid-line word mask contains bits that the line mask does not,
        // then the line mask is shorter than the mid-line word mask, and thus
        // the current word terminates on that newline.  If any such bits exist,
        // the MSB will be one of them, thus it is a negative integer.

        // Mark bytes in the same word (ignoring newlines).
        let mlw = self.mid_line_word_mask(view);

        // Mark bytes in the same line.
        let l = self.line_mask(view);

        // Mark bytes that are not on this line (including line terminator).
        let nl = _mm256_andnot_si256(l, mlw);

        // Check that at least one byte is not on this line.
        // If any bytes are not on this line, the MSB will be set.
        _mm256_cmpgt_epi64(_mm256_setzero_si256(), nl)
    }

    /// Whether the end of the word was found.
    ///
    /// This is all-ones if whitespace was found in the current block, as this
    /// delimits the current token.
    ///
    /// Latency (view -> return): 7c
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn end_of_word(&self, view: __m256i) -> __m256i {
        // If the MSB is not set, the word mask did not include the last byte.
        _mm256_cmpgt_epi64(self.word_mask(view), _mm256_set1_epi64x(-1))
    }

    /// A mask of the current word.
    ///
    /// Bytes are all-ones if they are within the current word (excluding the
    /// terminating whitespace).
    ///
    /// Latency (view -> return): 6c
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn word_mask(&self, view: __m256i) -> __m256i {
        // Truncate the mid-line word mask if a newline is within it.
        _mm256_and_si256(self.mid_line_word_mask(view), self.line_mask(view))
    }

    /// A mask of the current line.
    ///
    /// Bytes are all-ones if they are within the current line (excluding the
    /// line terminator itself).
    ///
    /// Latency (view -> return): 4c
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn line_mask(&self, view: __m256i) -> __m256i {
        // Mark ASCII newlines (0x0A).
        let lf = _mm256_cmpeq_epi8(view, _mm256_set1_epi8(b'\n' as i8));

        // Mark newline bytes by their LSB.
        let lf = _mm256_and_si256(lf, _mm256_set1_epi8(1));

        // Toggle bytes up to (and including) the first newline.
        let mask = _mm256_sub_epi64(lf, _mm256_set1_epi64x(1));

        // Mark all bytes up to (but excluding) the first newline.
        _mm256_andnot_si256(lf, mask)
    }

    /// A mask of the current word, assuming newlines are absent.
    ///
    /// Bytes are all-ones if they are within the current word (excluding the
    /// terminating whitespace).  Newlines are not considered as terminators,
    /// instead being treated as parts of words.
    ///
    /// Latency (view -> return): 5c
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mid_line_word_mask(&self, view: __m256i) -> __m256i {
        // Mark mid-line spaces.
        let sp = self.mid_line_space(view);

        // Mark mid-line bytes by their LSB.
        let sp = _mm256_and_si256(sp, _mm256_set1_epi8(1));

        // Toggle bytes up to (and including) the first mid-line space.
        let mask = _mm256_sub_epi64(sp, _mm256_set1_epi64x(1));

        // Mark all bytes up to (but excluding) the first mid-line space.
        _mm256_andnot_si256(sp, mask)
    }

    /// A mask of mid-line whitespace (spaces and tabs).
    ///
    /// Bytes are all-ones if they are ASCII spaces ('0x20') or horizontal tabs
    /// ('0x09'), and are otherwise all-zeros.
    ///
    /// Latency (view -> return): 2c
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mid_line_space(&self, view: __m256i) -> __m256i {
        // Mark ASCII spaces (0x20).
        let sp = _mm256_cmpeq_epi8(view, _mm256_set1_epi8(b'=' as i8));

        // Mark ASCII horizontal tabs (0x09).
        let ht = _mm256_cmpeq_epi8(view, _mm256_set1_epi8(b'\t' as i8));

        // Combine marks.
        _mm256_or_si256(sp, ht)
    }

    /// The current view into the segment.
    ///
    /// The view consists of the next 8 bytes of the segment.  If the segment's
    /// work has been completed, all-zeros is returned.
    ///
    /// Latency (offset -> return): 16c
    /// Latency (working -> return): 15c
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn view(&self, offset: __m256i, working: __m256i) -> __m256i {
        // The value returned if the segment is not working.
        let fallback = _mm256_setzero_si256();

        // The base address the segment is loaded from.
        let addr = self.input.as_ptr() as *const i64;

        // Load the segment's bytes if it is working.
        _mm256_mask_i64gather_epi64(fallback, addr, offset, working, 1)
    }

    /// Divide the frame into segments.
    ///
    /// A segment is a slice of the frame that begins with a newline.  This is
    /// important since lexing cannot start in the middle of a record.
    ///
    /// Initially, 4 segments are drawn by dividing the frame equally.  Each of
    /// the frames is then skipped forward until a newline.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn segment(&self) -> Result<(__m256i, __m256i), Error> {
        // The offset of the segment within the frame.
        let mut offset = _mm256_set_epi64x(4, 3, 2, 1);
        offset = _mm256_slli_epi64(offset, 16 - 2 - 3);
        offset = _mm256_sub_epi64(offset, _mm256_set1_epi64x(1));

        // The first 8 bytes in the segment.
        let mut input = _mm256_setzero_si256();

        // A mask of newlines in the input.
        let mut lines = _mm256_setzero_si256();

        // Whether a newline needs to be searched for.
        let mut search = _mm256_set1_epi64x(!0);

        // Search for a newline.
        //
        // Latency (offset -> input): 16c
        // Latency (search -> input): 15c
        // Latency (input -> search): 4c
        // Latency (input -> next offset): 4c
        // Latency (offset -> next offset): 1c
        // Latency (input -> break): 14c
        for _ in 0..2048 {
            // Load the next 8 bytes of the segment.
            input = _mm256_mask_i64gather_epi64(
                input,
                self.input.as_ptr() as *const _,
                offset,
                search,
                8,
            );

            // Mark newline bytes (all bits set if byte is equal).
            lines = _mm256_cmpeq_epi8(input, _mm256_set1_epi8(b'\n' as _));

            // Check whether the search needs to continue.
            search = _mm256_cmpeq_epi64(lines, _mm256_setzero_si256());

            // Push the offset backward if the search needs to continue.
            offset = _mm256_add_epi64(offset, search);

            // If all searches are finished, stop.
            if _mm256_testz_si256(search, search) != 0 {
                break;
            }
        }

        // If a newline wasn't found, return an error.
        if _mm256_testz_si256(search, search) == 0 {
            return Err(Error);
        }

        // Mark newline bytes (MSB is set for newlines).
        lines = _mm256_and_si256(lines, _mm256_set1_epi8(!0x7F));

        // Switch the offsets to units of bytes.
        offset = _mm256_slli_epi64(offset, 3);

        // Select all bytes up to (and including) the first newline byte.
        let to_skip = _mm256_sub_epi64(lines, _mm256_set1_epi64x(1));
        let to_skip = _mm256_xor_si256(lines, to_skip);

        // Update the offset to skip past the newline.
        offset = _mm256_add_epi64(offset, Self::count_bytes(to_skip));

        // Calculate the start of each segment.
        let start = _mm256_permute4x64_epi64(offset, 0b10010011);
        let start = _mm256_and_si256(start, _mm256_set_epi64x(!0, !0, !0, 0));

        Ok((start, offset))
    }

    /// Count the number of set bytes.
    ///
    /// The input must contain bytes that are all-ones or all-zeros.  The total
    /// number of all-ones bytes (between 0 and 8, inclusive) is returned.
    ///
    /// Latency (bytes -> result): 8c
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn count_bytes(mut bytes: __m256i) -> __m256i {
        // Negate bytes so that each one is 0 or 1.
        bytes = _mm256_sub_epi8(_mm256_setzero_si256(), bytes);

        // Pair bytes in the low and high dwords and add them.
        bytes = _mm256_add_epi8(bytes, _mm256_srli_epi64(bytes, 32));

        // Pair bytes in the low and high words and add them.
        bytes = _mm256_add_epi8(bytes, _mm256_srli_epi64(bytes, 16));

        // Pair the low and high bytes and add them.
        bytes = _mm256_add_epi8(bytes, _mm256_srli_epi64(bytes, 8));

        // Mask out intermediate results.
        bytes = _mm256_and_si256(bytes, _mm256_set1_epi64x(0xFF));

        bytes
    }
}

/// A lexical entry.
///
/// Entries are lines in a zone file providing DNS information.  This type is an
/// efficient and compact representation for an entry that can be produced by a
/// vectorized lexer.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
struct Entry {
    /// The address of this record.
    ///
    /// This is an offset in bytes relative to the frame.
    addr: u32,

    /// The location of the record data.
    ///
    /// The most-significant bit is set if an explicit domain name was present
    /// in the record.
    ///
    /// The remaining bits are an offset in bytes from the start of the record.
    data: u16,

    /// The location of the TTL.
    ///
    /// This is an offset in bytes from the start of the record.  If it is zero,
    /// an implicit TTL was used.
    ttl: u8,

    /// The record type.
    ///
    /// This is a unique identifier for a record or control entry type, derived
    /// by a perfect hash of the first few bytes of its name.
    hash: u8,
}

#[derive(Copy, Clone, Debug)]
struct Error;

fn main() -> io::Result<()> {
    let path = env::args().nth(1).unwrap();
    let mut input = Vec::with_capacity(4096 * 16 + 8);
    File::open(path)?.take(4096 * 16).read_to_end(&mut input)?;
    input.resize(input.capacity(), 0u8);

    let mut entries = vec![Entry::default(); 512 * 16];

    let frame = Frame {
        input: (&input[..65536]).try_into().unwrap(),
        entries: (&mut entries[..8192]).try_into().unwrap(),
    };

    unsafe { frame.lex() }.unwrap();

    Ok(())
}
