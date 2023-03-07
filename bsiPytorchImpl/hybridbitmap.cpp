#ifndef HYBRIDBITMAP_H
#define HYBRIDBITMAP_H

#include <algorithm>
#include <vector>
#include <queue>

#include "hybridutil.cpp"
#include "boolarray.cpp"
#include "runninglengthword.cpp"

template <class uword> class HybridBitmapIterator;

template <class uword> class HybridBitmapSetBitForwardIterator;

class BitmapStatistics;

template <class uword> class HybridBitmapRawIterator;

/**
 * This class is a compressed bitmap.
 * This is where compression
 * happens.
 * The underlying data structure is an STL vector.
 */
/*
###################################################################################################################
 */
template <class uword = uint32_t> class HybridBitmap {
public:
    
    double density=0;
    bool verbatim = false;
    std::vector<uword> buffer;
//    std::vector<uword>& buffer = *buffer1;
    
    
    
    HybridBitmap() : buffer(1, 0), sizeinbits(0), lastRLW(0) {}
    /**
     * @param buffersize :allocates the buffer size with all zero values
     */
    HybridBitmap(size_t buffersize) : buffer(buffersize, 0), sizeinbits(0), lastRLW(0) {}

    HybridBitmap(bool verbatim) : buffer(1, 0), sizeinbits(0), lastRLW(0), verbatim(verbatim) {}

    HybridBitmap(bool verbatim, size_t buffersize) : buffer(buffersize, 0), sizeinbits(buffersize*wordinbits), lastRLW(0), verbatim(verbatim) {}
    
    HybridBitmap(const HybridBitmap &other)
    : buffer(other.buffer), sizeinbits(other.sizeinbits),
    lastRLW(other.lastRLW),verbatim(other.verbatim), density(other.density) {}

    /**
     * Move constructor.
     */
    HybridBitmap(HybridBitmap &&other)
    : buffer(other.buffer), sizeinbits(other.sizeinbits),
    lastRLW(other.lastRLW),verbatim(other.verbatim), density(other.density)  {}


//    static HybridBitmap bitmapOf(size_t n, ...) {
//
//        HybridBitmap ans;
//        va_list vl;
//        va_start(vl, n);
//        for (size_t i = 0; i < n; i++) {
//            ans.set(static_cast<size_t>(va_arg(vl, int)));
//        }
//        va_end(vl);
//        return ans;
//    }
    static HybridBitmap bitmapOf(size_t n, ...) {
        HybridBitmap ans;
        va_list vl;
        va_start(vl, n);
        for (size_t i = 0; i < n; i++) {
            ans.set(static_cast<size_t>(va_arg(vl, int)));
        }
        va_end(vl);
        return ans;
    }

    static HybridBitmap verbatimBitmapOf(size_t n, ...) {
        
        HybridBitmap ans;
        va_list vl;
        va_start(vl, n);
        std::vector<size_t>vectorOfValues(0);

        for (size_t i = 0; i < n; i++) {
            vectorOfValues.push_back(static_cast<size_t>(va_arg(vl, int)));
        }
        va_end(vl);
        size_t wordSize = vectorOfValues.back()>>6;
        ans.buffer.resize(wordSize + 1);
        
        for (size_t i = 0; i < n; i++) {
            long position = vectorOfValues[i]>>6;
            long word = 1l << (vectorOfValues[i] % wordinbits);
            ans.buffer[position] = ans.buffer[position] | word;
        }
        ans.setSizeInBits(vectorOfValues.back() + 1);
        double aaa = n;
        double a1 = aaa/ans.sizeinbits;
        ans.setDensity(a1);
        return ans;
    }


    /**
     * Recover wasted memory usage. Fit buffers to the actual data.
     */
    void trim() { buffer.shrink_to_fit(); }

    /**
     * Query the value of bit i. This runs in time proportional to
     * the size of the bitmap. This is not meant to be use in
     * a performance-sensitive context.
     *
     *  (This implementation is based on zhenjl's Go version of JavaEWAH.)
     *
     */
    bool get(const size_t pos) const {
        if (pos >= static_cast<size_t>(sizeinbits))
            return false;
        const size_t wordpos = pos / wordinbits;
        size_t WordChecked = 0;
        if(buffer.size() == 0){
            return false;
        }
        if(verbatim){
            uword word = buffer[wordpos];
            if(((word>>(pos%wordinbits))&1)!=0)
                return true;
            else return false;
        }
        HybridBitmapRawIterator<uword> j = raw_iterator();
        while (j.hasNext()) {
            BufferedRunningLengthWord<uword> &rle = j.next();
            WordChecked += static_cast<size_t>(rle.getRunningLength());
            if (wordpos < WordChecked)
                return rle.getRunningBit();
            if (wordpos < WordChecked + rle.getNumberOfLiteralWords()) {
                const uword w = j.dirtyWords()[wordpos - WordChecked];
                return (w & (static_cast<uword>(1) << (pos % wordinbits))) != 0;
            }
            WordChecked += static_cast<size_t>(rle.getNumberOfLiteralWords());
        }
        return false;
    }

    
    uword getWord(int index) const{
            return buffer[index];
    }
    
    void setWord(int index, uword value){
        if(verbatim){
            buffer[index] = value;
        }
    }
    /**
     * Returns true if no bit is set.
     */
    bool empty() const {
        size_t pointer(0);
        while (pointer < buffer.size()) {
            ConstRunningLengthWord<uword> rlw(buffer[pointer]);
            if (rlw.getRunningBit()) {
                if(rlw.getRunningLength() > 0) return false;
            }
            ++pointer;
            for (size_t k = 0; k < rlw.getNumberOfLiteralWords(); ++k) {
                if(buffer[pointer] != 0) return false;
                ++pointer;
            }
        }
        return true;
    }


    /**
     * Set the ith bit to true (starting at zero).
     * Auto-expands the bitmap. It has constant running time complexity.
     * Note that you must set the bits in increasing order:
     * set(1), set(2) is ok; set(2), set(1) is not ok.
     * set(100), set(100) is also not ok.
     *
     * Note: by design EWAH is not an updatable data structure in
     * the sense that once bit 1000 is set, you cannot change the value
     * of bits 0 to 1000.
     *
     * Returns true if the value of the bit was changed, and false otherwise.
     * (In practice, if you set the bits in strictly increasing order, it
     * should always return true.)
     */
    bool set(size_t i);

    /**
     * Transform into a string that presents a list of set bits.
     * The running time is linear in the compressed size of the bitmap.
     */
    operator std::string() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }
    friend std::ostream &operator<<(std::ostream &out, const HybridBitmap &a) {

        out << "{";
        for (HybridBitmap::const_iterator i = a.begin(); i != a.end();) {
            out << *i;
            ++i;
            if (i != a.end())
                out << ",";
        }
        out << "}";

        return out;
    }
    /**
     * Make sure the two bitmaps have the same size (padding with zeroes
     * if necessary). It has constant running time complexity.
     *
     * This is useful when calling "logicalnot" functions.
     *
     * This can an adverse effect of performance, especially when computing
     * intersections.
     */
    void makeSameSize(HybridBitmap &a) {
        if (a.sizeinbits < sizeinbits)
            a.padWithZeroes(sizeinbits);
        else if (sizeinbits < a.sizeinbits)
            padWithZeroes(a.sizeinbits);
    }

    enum { RESERVEMEMORY = true }; // for speed

    typedef HybridBitmapSetBitForwardIterator<uword> const_iterator;

    /**
     * Returns an iterator that can be used to access the position of the
     * set bits. The running time complexity of a full scan is proportional to the
     * number
     * of set bits: be aware that if you have long strings of 1s, this can be
     * very inefficient.
     *
     * It can be much faster to use the toArray method if you want to
     * retrieve the set bits.
     */
    const_iterator begin() const {
        return HybridBitmapSetBitForwardIterator<uword>(&buffer);
    }

    /**
     * Basically a bogus iterator that can be used together with begin()
     * for constructions such as for(HybridBitmap<uword>::iterator i = b.begin();
     * i!=b.end(); ++i) {}
     */
    const_iterator &end() const {
        return HybridBitmapSetBitForwardIterator<uword>::end();
    }

    /**
     * Retrieve the set bits. Can be much faster than iterating through
     * the set bits with an iterator.
     */
    std::vector<size_t> toArray() const;


    /**
     * computes the logical and with another bitmap
     * answer goes into container
     * Running time complexity is proportional to the sum of the bitmap sizes.
     *
     * The sizeInBits() of the result is equal to the maximum that of the current
     * bitmap's sizeInBits() and that of a.sizeInBits().
     */
    void And(const HybridBitmap &a, HybridBitmap &container) const;
    void Xor(const HybridBitmap &a, HybridBitmap &container) const;
    void XorInPlace(const HybridBitmap &a);
    void Or(const HybridBitmap &a, HybridBitmap &container) const;
    void Not(const HybridBitmap &container) const;
    void xorNot(const HybridBitmap &a, HybridBitmap &container) const;
    void orAndNotV(const HybridBitmap &a, const HybridBitmap &b, HybridBitmap &container) const;
    void orAndV(const HybridBitmap &a, const HybridBitmap &b, HybridBitmap &container) const;
    void andVerbatim(const HybridBitmap &a, const HybridBitmap &b, HybridBitmap &container) const;
    void andVerbatim(const HybridBitmap &a,HybridBitmap &container) const;
    void orVerbatim(const HybridBitmap &a, HybridBitmap &container) const;
    void andHybridCompress(const HybridBitmap &a, HybridBitmap &container) const;
    void orHybridCompress(const HybridBitmap &a, HybridBitmap &container) const;
    void xorVerbatim(const HybridBitmap &a, HybridBitmap &container)const;
    void inPlaceXorVerbatim(const HybridBitmap &a);
    void inPlaceXorHybrid(const HybridBitmap &a);
    void andNotHybridCompress(const HybridBitmap &a, HybridBitmap &container)const;
    void andVerbatimCompress(const HybridBitmap &a, HybridBitmap &container)const;
    void andNotVerbatimCompress(const HybridBitmap &a, HybridBitmap &container)const;
    void andHybrid(const HybridBitmap &a, HybridBitmap &container)const;
    void andNotHybrid(const HybridBitmap &a, HybridBitmap &container)const;
    void andNotVerbatim(const HybridBitmap &a, HybridBitmap &container)const;
    void xorNotVerbatim(const HybridBitmap &a, HybridBitmap &container)const;
    void orHybrid(const HybridBitmap &a, HybridBitmap &container)const;
    void xorVerbatimCompress(const HybridBitmap &a, HybridBitmap &container)const;
    void xorHybridCompress(const HybridBitmap &a, HybridBitmap &container)const;
    void xorNotHybrid(const HybridBitmap &a, HybridBitmap &container)const;
    void xorHybrid(const HybridBitmap &a, HybridBitmap &container)const;
    void logicalAndDecompress(const HybridBitmap &a, HybridBitmap &container)const;
    void logicalxorDecompress(const HybridBitmap &a, HybridBitmap &container)const;
    void logicalxornot(const HybridBitmap &a, HybridBitmap &container) const;
    void add(long newdata, const int bitsthatmatter);
    void add(const long newdata) ;
    void andNot(const HybridBitmap &a, HybridBitmap &container)const;
    void majInPlace(const HybridBitmap &a,const HybridBitmap &b);
    void selectMultiplication(const HybridBitmap &res,const HybridBitmap &FS, HybridBitmap &container)const;

    void shiftRow(const HybridBitmap &this_bitmap, HybridBitmap &other_bitmap);
    void shiftRowWithCarry(const HybridBitmap &this_bitmap, HybridBitmap &other_bitmap, HybridBitmap &carry);
    
    
    
    
    void AndInPlace(const HybridBitmap &a );
    void andVerbatimCompressInPlace(const HybridBitmap &a );
    void andVerbatimInPlace(const HybridBitmap &a );
    void andHybridCompressInPlace(const HybridBitmap &a );
    void logicalandInPlace(const HybridBitmap &a );
    void selectMultiplicationInPlace(const HybridBitmap &a, const HybridBitmap &FS);

    /**
     * computes the logical and with another compressed bitmap
     * answer goes into container
     * Running time complexity is proportional to the sum of the compressed
     * bitmap sizes.
     *
     * The sizeInBits() of the result is equal to the maximum that of the current
     * bitmap's sizeInBits() and that of a.sizeInBits().
     */
    void logicaland(const HybridBitmap &a, HybridBitmap &container) const;


    void maj(const HybridBitmap &a,const HybridBitmap &b, HybridBitmap &container) const;
    HybridBitmap maj(const HybridBitmap &a, const HybridBitmap &b) const {
        HybridBitmap answer;
        maj(a, b, answer);
        return answer;
    }
    /*
     * selectMultiplication perform following operation used in multiplication method.
     * container = ~it.res + it.FS
     *
     */
    HybridBitmap selectMultiplication(const HybridBitmap &res,const HybridBitmap &FS)const{
        HybridBitmap answer;
        selectMultiplication(res, FS, answer);
        return answer;
    }

//    HybridBitmap shiftRow(const HybridBitmap &res,const HybridBitmap &this_bitmap, HybridBitmap &other_bitmap)const{
//        HybridBitmap answer;
//        shiftRow(res, this_bitmap, other_bitmap);
//        return answer;
//    }
//
//    HybridBitmap shiftRowWithCarry(const HybridBitmap &res,const HybridBitmap &this_bitmap, HybridBitmap &other_bitmap, HybridBitmap &carry)const{
//        HybridBitmap answer;
//        selectMultiplication(res, FS, answer);
//        return answer;
//    }
    /**
     * computes the logical and with another  bitmap
     * Return the answer
     * Running time complexity is proportional to the sum of the bitmap sizes.
     *
     * The sizeInBits() of the result is equal to the maximum that of the current
     * bitmap's sizeInBits() and that of a.sizeInBits().
     */
    
    HybridBitmap And(const HybridBitmap &a) const {
        HybridBitmap answer;
        And(a, answer);
        return answer;
    }

    HybridBitmap Or(const HybridBitmap &a) const {
        HybridBitmap answer;
        Or(a, answer);
        return answer;
    }
    
    
    HybridBitmap orVerbatimCompress(const HybridBitmap &a) const {
        HybridBitmap answer;
        orVerbatimCompress(a, answer);
        return answer;
    }
    HybridBitmap Xor(const HybridBitmap &a) const {
        HybridBitmap answer;
        Xor(a, answer);
        return answer;
    }

    HybridBitmap xorNot(const HybridBitmap &a) const {
        HybridBitmap answer;
        xorNot(a, answer);
        return answer;
    }
    HybridBitmap orAndNotV(const HybridBitmap &a, const HybridBitmap &b) const {
        HybridBitmap answer;
        orAndNotV(a, b, answer);
        return answer;
    }
    HybridBitmap orAndV(const HybridBitmap &a, const HybridBitmap &b) const {
        HybridBitmap answer;
        orAndV(a, b, answer);
        return answer;
    }
    HybridBitmap andVerbatim(const HybridBitmap &a, const HybridBitmap &b) const {
        HybridBitmap answer;
        andVerbatim(a, b, answer);
        return answer;
    }
    HybridBitmap andVerbatim(const HybridBitmap &a) const {
        HybridBitmap answer;
        andVerbatim(a, answer);
        return answer;
    }
    HybridBitmap orVerbatim(const HybridBitmap &a) const {
        HybridBitmap answer;
        orVerbatim(a, answer);
        return answer;
    }

    HybridBitmap andHybridCompress(const HybridBitmap &a) const {
        HybridBitmap answer;
        andHybridCompress(a, answer);
        return answer;
    }
    HybridBitmap orHybrid(const HybridBitmap &a) const {
        HybridBitmap answer;
        orHybrid(a, answer);
        return answer;
    }
    HybridBitmap orHybridCompress(const HybridBitmap &a) const {
        HybridBitmap answer;
        orHybridCompress(a, answer);
        return answer;
    }
    HybridBitmap andNotHybridCompress(const HybridBitmap &a) const {
        HybridBitmap answer;
        andNotHybridCompress(a, answer);
        return answer;
    }
    HybridBitmap andVerbatimCompress(const HybridBitmap &a)const{
        HybridBitmap answer;
        andVerbatimCompress(a, answer);
        return answer;
    }
    HybridBitmap andNotVerbatimCompress(const HybridBitmap &a)const{
        HybridBitmap answer;
        andNotVerbatimCompress(a, answer);
        return answer;
    }
    HybridBitmap andNotV(const HybridBitmap &a)const{
        HybridBitmap answer;
        andNotV(a, answer);
        return answer;
    }
    HybridBitmap andC(const HybridBitmap &a)const{
        HybridBitmap answer;
        answer.buffer.reserve(bufferSize() > a.bufferSize() ? bufferSize() : a.bufferSize());
        logicaland(a, answer);
        return answer;
    }
    HybridBitmap andNotC(const HybridBitmap &a)const{
        HybridBitmap answer;
        answer.buffer.reserve(bufferSize() > a.bufferSize() ? bufferSize() : a.bufferSize());
        logicalandnot(a, answer);
        return answer;
    }
    HybridBitmap andNotV(const HybridBitmap &a, HybridBitmap &container)const{
        HybridBitmap answer;
        answer.density = density*(1-a.density);
        andNotVerbatim(a, answer);
        return answer;
    }
    HybridBitmap andHybrid(const HybridBitmap &a)const{
        HybridBitmap answer;
        andHybrid(a, answer);
        return answer;
    }
    HybridBitmap andNotHybrid(const HybridBitmap &a)const{
        HybridBitmap answer;
        andNotHybrid(a, answer);
        return answer;
    }
    HybridBitmap andNotVerbatim(const HybridBitmap &a)const{
        HybridBitmap answer;
        andNotVerbatim(a, answer);
        return answer;
    }
    HybridBitmap xorC(const HybridBitmap &a)const  {
        HybridBitmap container = new HybridBitmap();
        container.reserve(bufferSize() + a.bufferSize());
        logicalxor(a, container);
        return container;
    }
    double getDensity(){
        return density;
    }
    
    HybridBitmap xorVerbatim(const HybridBitmap &a)const {
        HybridBitmap container;
        xorVerbatim(a, container);
        return container;
    }

    HybridBitmap xorVerbatimCompress(const HybridBitmap &a)const {
        HybridBitmap container;
        xorVerbatimCompress(a, container);
        return container;
    }
    
    HybridBitmap xorNotVerbatim(const HybridBitmap &a)const {
        HybridBitmap container;
        xorVerbatimCompress(a, container);
        return container;
    }
    
    HybridBitmap xorNotHybrid(const HybridBitmap &a)const {
        HybridBitmap container;
        xorVerbatimCompress(a, container);
        return container;
    }
    
    
    HybridBitmap xorHybridCompress(const HybridBitmap &a)const {
        HybridBitmap container;
        xorVerbatimCompress(a, container);
        return container;
    }
    

    HybridBitmap xorHybrid(const HybridBitmap &a)const {
        HybridBitmap container;
        xorHybrid(a, container);
        return container;
    }
    
    HybridBitmap Not()const {
        HybridBitmap container;
        container = logicalnot();
        return container;
    }
    
    
    bool compareBitmap(const HybridBitmap &a)const;
    /**
     * Operates two verbatim vectors and the result is compressed
     * @param, a
     * @param, container
     */
    //void andVerbatimCompress(const HybridBitmap &a, HybridBitmap &container) const;
    void orVerbatimCompress(const HybridBitmap &a, HybridBitmap &container) const;
    void notHybridBitmap();
    /**
     * computes the logical and with another compressed bitmap
     * Return the answer
     * Running time complexity is proportional to the sum of the compressed
     * bitmap sizes.
     *
     * The sizeInBits() of the result is equal to the maximum that of the current
     * bitmap's sizeInBits() and that of a.sizeInBits().
     */
    HybridBitmap logicaland(const HybridBitmap &a) const {
        HybridBitmap answer;
        logicaland(a, answer);
        return answer;
    }

    /**
     * calls logicaland
     */
    HybridBitmap operator&(const HybridBitmap &a) const {
        return And(a);
    }

    /**
     * computes the logical and-not with another bitmap answer goes into container
     * Running time complexity is proportional to the sum of the bitmap sizes.
     *
     * The sizeInBits() of the result should be equal to that of the current
     * bitmap irrespective of a.sizeInBits().
     *
     */
    //void andNot(const HybridBitmap &a, HybridBitmap &container) const;

    /**
     * computes the logical and with another compressed bitmap
     * answer goes into container
     * Running time complexity is proportional to the sum of the compressed
     * bitmap sizes.
     *
     * The sizeInBits() of the result should be equal to that of the current
     * bitmap irrespective of a.sizeInBits().
     *
     */
    void logicalandnot(const HybridBitmap &a, HybridBitmap &container) const;



    /**
     * calls logicalandnot
     */
    HybridBitmap operator-(const HybridBitmap &a) const {
        return logicalnot(a);
    }

    /**
     * computes the logical and-not with another compressed bitmap
     * Return the answer
     * Running time complexity is proportional to the sum of the compressed
     * bitmap sizes.
     *
     * The sizeInBits() of the result should be equal to that of the current
     * bitmap irrespective of a.sizeInBits().
     *
     */
    HybridBitmap andNot(const HybridBitmap &a) const {
        HybridBitmap answer;
        andNot(a, answer);
        return answer;
    }

    /**
     * computes the logical and-not with another compressed bitmap
     * Return the answer
     * Running time complexity is proportional to the sum of the compressed
     * bitmap sizes.
     *
     * The sizeInBits() of the result should be equal to that of the current
     * bitmap irrespective of a.sizeInBits().
     *
     */
    HybridBitmap logicalandnot(const HybridBitmap &a) const {
        HybridBitmap answer;
        logicalandnot(a, answer);
        return answer;
    }

    /**
     * tests whether the bitmaps "intersect" (have at least one 1-bit at the same
     * position). This function does not modify the existing bitmaps.
     * It is faster than calling logicaland.
     */
    bool intersects(const HybridBitmap &a) const;

    /**
     * computes the logical or with another bitmap
     * answer goes into container
     *
     *
     * If you have many bitmaps, see fast_logicalor_tocontainer.
     *
     * The sizeInBits() of the result is equal to the maximum that of the current
     * bitmap's sizeInBits() and that of a.sizeInBits().
     */


    /**
     * computes the logical or with another bitmap
     * answer goes into container
     * Running time complexity is proportional to the sum of the bitmap sizes.
     *
     * If you have many bitmaps, see fast_logicalor_tocontainer.
     *
     * The sizeInBits() of the result is equal to the maximum that of the current
     * bitmap's sizeInBits() and that of a.sizeInBits().
     */
    void logicalor(const HybridBitmap &a, HybridBitmap &container) const;

    /**
     * computes the logical or with another compressed bitmap
     * answer goes into a non compressed container
     * Running time complexity is proportional to the sum of the compressed
     * bitmap sizes.
     *
     *
     * The sizeInBits() of the result is equal to the maximum that of the current
     * bitmap's sizeInBits() and that of a.sizeInBits().
     */
    void logicalorDecompress(const HybridBitmap &a,  HybridBitmap &container) const;
    
    /**
     * computes the size (in number of set bits) of the logical or with another
     * compressed bitmap
     * Running time complexity is proportional to the sum of the compressed
     * bitmap sizes.
     */
    size_t logicalorcount(const HybridBitmap &a) const;

    /**
     * computes the size (in number of set bits) of the logical and with another
     * compressed bitmap
     * Running time complexity is proportional to the sum of the compressed
     * bitmap sizes.
     */
    size_t logicalandcount(const HybridBitmap &a) const;

    /**
     * computes the size (in number of set bits) of the logical and not with
     * another compressed bitmap
     * Running time complexity is proportional to the sum of the compressed
     * bitmap sizes.
     */
    size_t logicalandnotcount(const HybridBitmap &a) const;

    /**
     * computes the size (in number of set bits) of the logical xor with another
     * compressed bitmap
     * Running time complexity is proportional to the sum of the compressed
     * bitmap sizes.
     */
    size_t logicalxorcount(const HybridBitmap &a) const;

    /**
     * computes the logical or with another compressed bitmap
     * Return the answer
     * Running time complexity is proportional to the sum of the compressed
     * bitmap sizes.
     *
     * If you have many bitmaps, see fast_logicalor.
     *
     * The sizeInBits() of the result is equal to the maximum that of the current
     * bitmap's sizeInBits() and that of a.sizeInBits().
     */
    HybridBitmap logicalor(const HybridBitmap &a) const {
        HybridBitmap answer;
        logicalor(a, answer);
        return answer;
    }
    
    HybridBitmap logicalxornot(const HybridBitmap &a) const {
        HybridBitmap answer;
        logicalxornot(a, answer);
        return answer;
    }
    /**
     * calls logicalor
     */
    HybridBitmap operator|(const HybridBitmap &a) const { return logicalor(a); }

    /**
     * computes the logical xor with another compressed bitmap
     * answer goes into container
     * Running time complexity is proportional to the sum of the compressed
     * bitmap sizes.
     *
     * The sizeInBits() of the result is equal to the maximum that of the current
     * bitmap's sizeInBits() and that of a.sizeInBits().
     */
    void logicalxor(const HybridBitmap &a, HybridBitmap &container) const;

    /**
     * computes the logical xor with another compressed bitmap
     * Return the answer
     * Running time complexity is proportional to the sum of the compressed
     * bitmap sizes.
     *
     * The sizeInBits() of the result is equal to the maximum that of the current
     * bitmap's sizeInBits() and that of a.sizeInBits().
     */
    HybridBitmap logicalxor(const HybridBitmap &a) const {
        HybridBitmap answer;
        logicalxor(a, answer);
        return answer;
    }

    /**
     * calls logicalxor
     */
    HybridBitmap operator^(const HybridBitmap &a) const {
        return logicalxor(a);
    }
    /**
     * clear the content of the bitmap. It does not
     * release the memory.
     */
    void reset() {
        buffer.clear();
        //buffer.push_back(0);
        sizeinbits = 0;
        density = 0;
        lastRLW = 0;
    }

    /**
     * convenience method.
     *
     * returns the number of words added (storage cost increase)
     */
    inline size_t addWord(const uword newdata,
                          const uword bitsthatmatter = 8 * sizeof(uword));
    /**
     * Adds a word to a verbatim bitmap
     * @para newdata
     * @para bitsthatmatter
     */
    inline void addVerbatim(const uword newdata,
                            const uword bitsthatmatter);
    void addVerbatim(const uword newdata);

    inline void printout(std::ostream &o = std::cout) {
        //toBoolArray().printout(o);
        if(verbatim){
            for (size_t k = 0; k < sizeinbits; ++k){
                bool value = ( buffer[k / wordinbits] & (static_cast<uword>(1) << (k % wordinbits)) )!= 0;
                o << value << " ";
            }
            o << std::endl;
        }else{
            toBoolArray().printout(o);
        }
        
    }

    /**
     * Prints a verbose description of the content of the compressed bitmap.
     */
    void debugprintout() const;

    /**
     * Checks if this is a verbatim bitmap
     */
    inline bool isVerbatim() const { return verbatim; }

    /**
     * Return the size in bits of this bitmap (this refers
     * to the uncompressed size in bits).
     *
     * You can increase it with padWithZeroes()
     */
    inline size_t sizeInBits() const { return sizeinbits; }

    /**
     * Return the size of the buffer in bytes. This
     * is equivalent to the storage cost, minus some overhead.
     * See sizeOnDisk to get the actual storage cost with overhead.
     */
    inline size_t sizeInBytes() const { return buffer.size() * sizeof(uword); }

    /**
     * same as addEmptyWord, but you can do several in one shot!
     * returns the number of words added (storage cost increase)
     */
    size_t addStreamOfEmptyWords(const bool v, size_t number);

    /**
     * add a stream of dirty words, returns the number of words added
     * (storage cost increase)
     */
    size_t addStreamOfDirtyWords(const uword *v, const size_t number);

    /**
     * add a stream of dirty words, each one negated, returns the number of words
     * added
     * (storage cost increase)
     */
    size_t addStreamOfNegatedDirtyWords(const uword *v, const size_t number);

    /**
     * make sure the size of the array is totalbits bits by padding with zeroes.
     * returns the number of words added (storage cost increase).
     *
     * This is useful when calling "logicalnot" functions.
     *
     * This can an adverse effect of performance, especially when computing
     * intersections.
     *
     */
    size_t padWithZeroes(const size_t totalbits);

    /**
     * Compute the size on disk assuming that it was saved using
     * the method "write".
     */
    size_t sizeOnDisk(const bool savesizeinbits = true) const;

    /**
     * Save this bitmap to a stream. The file format is
     * | sizeinbits | buffer lenth | buffer content|
     * the sizeinbits part can be omitted if "savesizeinbits=false".
     * Both sizeinbits and buffer length are saved using the size_t data
     * type which is typically a 32-bit unsigned integer for 32-bit CPUs
     * and a 64-bit unsigned integer for 64-bit CPUs.
     * Note that this format is machine-specific. Note also
     * that the word size is not saved. For robust persistent
     * storage, you need to save this extra information elsewhere.
     *
     * Returns how many bytes were handed out to the stream.
     */
    size_t write(std::ostream &out, const bool savesizeinbits = true) const;

    /**
     * same as write(std::ostream...), except that you provide a char pointer
     * and a "capacity" (in bytes). The function never writes at or beyond "out+capacity".
     * If the storage needed exceeds the
     * given capacity, the value zero is returned: it should be considered an error.
     * Otherwise, the number of bytes copied is returned.
     */
    size_t write(char * out, size_t capacity, const bool savesizeinbits = true) const;

    /**
     * This only writes the content of the buffer (see write()) method.
     * It is for advanced users.
     */
    void writeBuffer(std::ostream &out) const;

    /**
     * size (in words) of the underlying STL vector.
     */
    size_t bufferSize() const { return buffer.size(); }

    /**
     * this is the counterpart to the write method.
     * if you set savesizeinbits=false, then you are responsible
     * for setting the value fo the attribute sizeinbits (see method
     * setSizeInBits).
     *
     * Returns how many bytes were queried from the stream.
     */
    size_t read(std::istream &in, const bool savesizeinbits = true);


    /**
     * same as read(std::istream...), except that you provide a char pointer
     * and a "capacity" (in bytes). The function never reads at or beyond "in+capacity".
     * If the detected storage exceeds the  given capacity, the value zero is returned:
     * it should be considered an error.
     * Otherwise, the number of bytes read is returned.
     */
    size_t read(const char * in, size_t capacity, const bool savesizeinbits = true);

    /**
     * read the buffer from a stream, see method writeBuffer.
     * this is for advanced users.
     */
    void readBuffer(std::istream &in, const size_t buffersize);

    /**
     * We define two HybridBitmap as being equal if they have the same set bits.
     * Alternatively, B1==B2 if and only if cardinality(B1 XOR B2) ==0.
     */
    bool operator==(const HybridBitmap &x) const;

    /**
     * We define two HybridBitmap as being different if they do not have the same
     * set bits.
     * Alternatively, B1!=B2 if and only if cardinality(B1 XOR B2) >0.
     */
    bool operator!=(const HybridBitmap &x) const;

    bool operator==(const BoolArray<uword> &x) const;

    bool operator!=(const BoolArray<uword> &x) const;

    /**
     * Iterate over the uncompressed words.
     * Can be considerably faster than begin()/end().
     * Running time complexity of a full scan is proportional to the
     * uncompressed size of the bitmap.
     */
    HybridBitmapIterator<uword> uncompress() const;

    /**
     * To iterate over the compressed data.
     * Can be faster than any other iterator.
     * Running time complexity of a full scan is proportional to the
     * compressed size of the bitmap.
     */
    HybridBitmapRawIterator<uword> raw_iterator() const;

    /**
     * Appends the content of some other compressed bitmap
     * at the end of the current bitmap.
     */
    void append(const HybridBitmap &x);

    /**
     * For research purposes. This computes the number of
     * dirty words and the number of compressed words.
     */
    BitmapStatistics computeStatistics() const;

    /**
     * For convenience, this fully uncompresses the bitmap.
     * Not fast!
     */
    BoolArray<uword> toBoolArray() const;

    /**
     * Convert to a list of positions of "set" bits.
     * The recommended container is vector<size_t>.
     *
     * See also toArray().
     */
    template <class container>
    void appendRowIDs(container &out, const size_t offset = 0) const;

    /**
     * Convert to a list of positions of "set" bits.
     * The recommended container is vector<size_t>.
     * (alias for appendRowIDs).
     *
     * See also toArray().
     */
    template <class container>
    void appendSetBits(container &out, const size_t offset = 0) const {
        return appendRowIDs(out, offset);
    }

    /**
     * Returns a vector containing the position of the set
     * bits in increasing order. This just calls "toArray".
     */
    std::vector<size_t> toVector() const { return toArray(); }

    /**
     * Returns the number of bits set to the value 1.
     * The running time complexity is proportional to the
     * compressed size of the bitmap.
     *
     * This is sometimes called the cardinality.
     */
    size_t numberOfOnes() const;

    /**
     * Swap the content of this bitmap with another bitmap.
     * No copying is done. (Running time complexity is constant.)
     */
    void swap(HybridBitmap &x);

    const std::vector<uword> &getBuffer() const { return buffer; }

    enum { wordinbits = sizeof(uword) * 8 };

    /**
     * Please don't copy your bitmaps! The running time
     * complexity of a copy is the size of the compressed bitmap.
     **/
  
    /**
     * Copies the content of one bitmap onto another. Running time complexity
     * is proportional to the size of the compressed bitmap.
     * please, never hard-copy this object. Use the swap method if you must.
     */
    HybridBitmap &operator=(const HybridBitmap &x) {
        buffer = x.buffer;
        sizeinbits = x.sizeinbits;
        lastRLW = x.lastRLW;
        density = x.density;
        verbatim = x.verbatim;
        return *this;
    }

   
    /**
     * Move assignment operator.
     */
    HybridBitmap &operator=(HybridBitmap &&x) {
        buffer.resize(x.bufferSize()+1);
        buffer.swap(x.buffer);
        sizeinbits = x.sizeinbits;
        lastRLW = x.lastRLW;
        density = x.density;
        verbatim = x.verbatim;
        return *this;
    }

    /**
     * This is equivalent to the operator =. It is used
     * to keep in mind that assignment can be expensive.
     *
     *if you don't care to copy the bitmap (performance-wise), use this!
     */
    void expensive_copy(const HybridBitmap &x) {
        buffer = x.buffer;
        sizeinbits = x.sizeinbits;
        lastRLW = x.lastRLW;
    }




    /**
     * Write the logical not of this bitmap in the provided container.
     *
     * This function takes into account the sizeInBits value.
     * You may need to call "padWithZeroes" to adjust the sizeInBits.
     */
    void logicalnot(HybridBitmap &x) const;

    /**
     * Write the logical not of this bitmap in the provided container.
     *
     * This function takes into account the sizeInBits value.
     * You may need to call "padWithZeroes" to adjust the sizeInBits.
     */
    HybridBitmap<uword> logicalnot() const {
        HybridBitmap answer;
        logicalnot(answer);
        return answer;
    }

    /**
     * Apply the logical not operation on this bitmap.
     * Running time complexity is proportional to the compressed size of the
     *bitmap.
     * The current bitmap is not modified.
     *
     * This function takes into account the sizeInBits value.
     * You may need to call "padWithZeroes" to adjust the sizeInBits.
     **/
    void inplace_logicalnot();

    /**
     * set size in bits. This does not affect the compressed size. It
     * runs in constant time. This should not normally be used, except
     * as part of a deserialization process.
     */
    inline void setSizeInBits(const size_t size) { sizeinbits = size; }
    
    inline void setDensity(const double newDensity){density = newDensity;}
    bool setSizeInBits(const size_t size, const bool defaultvalue) {
        if (size <= sizeinbits)
            return false;
        if((sizeinbits % wordinbits) != 0){
            long fillGap =0l;
            if(defaultvalue){
                fillGap = (1l<<(sizeinbits % wordinbits))-1;
            }
            
            sizeinbits += (sizeinbits % wordinbits);
            addLiteralWord(fillGap);
        }
        
        addStreamOfEmptyWords(defaultvalue, (size / wordinbits) - sizeinbits / wordinbits);
        // next bit could be optimized
        if(sizeinbits<size){
            long newdata =0l;
            if(defaultvalue){
                newdata = (1l<<(size - sizeinbits))-1;
            }
            sizeinbits += sizeof(uword)*8;
            addLiteralWord(newdata);
        }
        return true;
    }


    /**
     * Like addStreamOfEmptyWords but
     * addStreamOfEmptyWords but does not return the cost increase,
     * does not update sizeinbits
     */
    inline void fastaddStreamOfEmptyWords(const bool v, size_t number);
    /**
     * LikeaddStreamOfDirtyWords but does not return the cost increse,
     * does not update sizeinbits.
     */
    inline void fastaddStreamOfDirtyWords(const uword *v, const size_t number);

    // private because does not increment the size in bits
    // returns the number of words added (storage cost increase)
    inline size_t addLiteralWord(const uword newdata);
    
    // private because does not increment the size in bits
    // returns the number of words added (storage cost increase)
    size_t addEmptyWord(const bool v);
    // this second version "might" be faster if you hate OOP.
    // in my tests, it turned out to be slower!
    // private because does not increment the size in bits
    // inline void addEmptyWordStaticCalls(bool v);
    // RunningLengthWord<> rlw;

private:
    
    
    size_t sizeinbits;
    size_t lastRLW;

    double andThreshold =  0.0005;
    double orThreshold = 0.001;
};
/*
###################################################################################################################
*/
 /**
 * computes the logical or (union) between "n" bitmaps (referenced by a
 * pointer).
 * The answer gets written out in container. This might be faster than calling
 * logicalor n-1 times.
 */
template <class uword>
void fast_logicalor_tocontainer(size_t n, const HybridBitmap<uword> **inputs,
                                HybridBitmap<uword> &container);

/**
 * computes the logical or (union) between "n" bitmaps (referenced by a
 * pointer).
 * Returns the answer. This might be faster than calling
 * logicalor n-1 times.
 */
template <class uword>
HybridBitmap<uword> fast_logicalor(size_t n,
                                   const HybridBitmap<uword> **inputs) {
    HybridBitmap<uword> answer;
    fast_logicalor_tocontainer(n, inputs, answer);
    return answer;
}

/**
 * Iterate over words of bits from a compressed bitmap.
 */
/*
###################################################################################################################
*/
template <class uword> class HybridBitmapIterator {
public:
    /**
     * is there a new word?
     */
    bool hasNext() const { return pointer < myparent.size(); }

    /**
     * return next word.
     */
    uword next() {
        uword returnvalue;
        if (compressedwords < rl) {
            ++compressedwords;
            if (b)
                returnvalue = notzero;
            else
                returnvalue = zero;
        } else {
            ++literalwords;
            ++pointer;
            returnvalue = myparent[pointer];
        }
        if ((compressedwords == rl) && (literalwords == lw)) {
            ++pointer;
            if (pointer < myparent.size())
                readNewRunningLengthWord();
        }
        return returnvalue;
    }

    HybridBitmapIterator(const HybridBitmapIterator<uword> &other)
    : pointer(other.pointer), myparent(other.myparent),
    compressedwords(other.compressedwords),
    literalwords(other.literalwords), rl(other.rl), lw(other.lw),
    b(other.b) {}

    static const uword zero = 0;
    static const uword notzero = static_cast<uword>(~zero);

private:
    HybridBitmapIterator(const std::vector<uword> &parent);
    void readNewRunningLengthWord();
    friend class HybridBitmap<uword>;
    size_t pointer;
    const std::vector<uword> &myparent;
    uword compressedwords;
    uword literalwords;
    uword rl, lw;
    bool b;
};
/*
###################################################################################################################
*/

 /**
 * Used to go through the set bits. Not optimally fast, but convenient.
 */

/*
###################################################################################################################
 */
template <class uword> class HybridBitmapSetBitForwardIterator {
public:
    typedef std::forward_iterator_tag iterator_category;
    typedef size_t *pointer;
    typedef size_t &reference_type;
    typedef size_t value_type;
    typedef ptrdiff_t difference_type;
    typedef HybridBitmapSetBitForwardIterator<uword> type_of_iterator;
    /**
     * Provides the location of the set bit.
     */
    inline size_t operator*() const { return answer; }

    bool operator<(const type_of_iterator &o) const {
        if (!o.hasValue)
            return true;
        if (!hasValue)
            return false;
        return answer < o.answer;
    }

    bool operator<=(const type_of_iterator &o) const {
        if (!o.hasValue)
            return true;
        if (!hasValue)
            return false;
        return answer <= o.answer;
    }

    bool operator>(const type_of_iterator &o) const { return !((*this) <= o); }

    bool operator>=(const type_of_iterator &o) const { return !((*this) < o); }

    HybridBitmapSetBitForwardIterator &operator++() { //++i
        if (hasNext)
            next();
        else
            hasValue = false;
        return *this;
    }

    HybridBitmapSetBitForwardIterator operator++(int) { // i++
        HybridBitmapSetBitForwardIterator old(*this);
        if (hasNext)
            next();
        else
            hasValue = false;
        return old;
    }

    bool operator==(const HybridBitmapSetBitForwardIterator<uword> &o) const {
        if ((!hasValue) && (!o.hasValue))
            return true;
        return (hasValue == o.hasValue) && (answer == o.answer);
    }

    bool operator!=(const HybridBitmapSetBitForwardIterator<uword> &o) const {
        return !(*this == o);
    }

    static HybridBitmapSetBitForwardIterator<uword> &end() {
        static HybridBitmapSetBitForwardIterator<uword> e;
        return e;
    }

    HybridBitmapSetBitForwardIterator(const std::vector<uword> *parent,
                                      size_t startpointer = 0)
    : word(0), position(0), runningLength(0), literalPosition(0),
    wordPosition(startpointer), wordLength(0), buffer(parent),
    hasNext(false), hasValue(false), answer(0) {
        if (wordPosition < buffer->size()) {
            setRunningLengthWord();
            hasNext = moveToNext();
            if (hasNext) {
                next();
                hasValue = true;
            }
        }
    }

    HybridBitmapSetBitForwardIterator()
    : word(0), position(0), runningLength(0), literalPosition(0),
    wordPosition(0), wordLength(0), buffer(NULL), hasNext(false),
    hasValue(false), answer(0) {}

    inline bool runningHasNext() const { return position < runningLength; }

    inline bool literalHasNext() {
        while (word == 0 && wordPosition < wordLength) {
            word = (*buffer)[wordPosition++];
            literalPosition = position;
            position += WORD_IN_BITS;
        }
        return word != 0;
    }

    inline void setRunningLengthWord() {
        uword rlw = (*buffer)[wordPosition];
        runningLength =
        (size_t)WORD_IN_BITS * RunningLengthWord<uword>::getRunningLength(rlw) +
        position;
        if (!RunningLengthWord<uword>::getRunningBit(rlw)) {
            position = runningLength;
        }
        wordPosition++; // point to first literal word
        wordLength =
        wordPosition + RunningLengthWord<uword>::getNumberOfLiteralWords(rlw);
    }

    inline bool moveToNext() {
        while (!runningHasNext() && !literalHasNext()) {
            if (wordPosition >= buffer->size()) {
                return false;
            }
            setRunningLengthWord();
        }
        return true;
    }

    void next() { // update answer
        if (runningHasNext()) {
            answer = position++;
            if (runningHasNext())
                return;
        } else {
            uword t = word & (~word + 1);
            answer = literalPosition + countOnes((uword)(t - 1));
            word ^= t;
        }
        hasNext = moveToNext();
    }

    enum { WORD_IN_BITS = sizeof(uword) * 8 };
    uword word; // lit word
    size_t position;
    size_t runningLength;
    size_t literalPosition;
    size_t wordPosition; // points to word in buffer
    uword wordLength;
    const std::vector<uword> *buffer;
    bool hasNext;
    bool hasValue;
    size_t answer;
};
/*
 ##################################################################################################################
 */

/**
 * This object is returned by the compressed bitmap as a
 * statistical descriptor.
 */
/*
###################################################################################################################
 */
class BitmapStatistics {
public:
    BitmapStatistics()
    : totalliteral(0), totalcompressed(0), runningwordmarker(0),
    maximumofrunningcounterreached(0) {}
    size_t getCompressedSize() const { return totalliteral + runningwordmarker; }
    size_t getUncompressedSize() const { return totalliteral + totalcompressed; }
    size_t getNumberOfDirtyWords() const { return totalliteral; }
    size_t getNumberOfCleanWords() const { return totalcompressed; }
    size_t getNumberOfMarkers() const { return runningwordmarker; }
    size_t getOverRuns() const { return maximumofrunningcounterreached; }
    size_t totalliteral;
    size_t totalcompressed;
    size_t runningwordmarker;
    size_t maximumofrunningcounterreached;
};
/*
###################################################################################################################
 */



/**
 * This is a low-level iterator.
 */
/*
###################################################################################################################
 */
template <class uword = uint32_t> class HybridBitmapRawIterator {
public:
    HybridBitmapRawIterator(const HybridBitmap<uword> &p)
    : pointer(0), myparent(&p.getBuffer()), rlw((*myparent)[pointer], this) {}
    HybridBitmapRawIterator(const HybridBitmapRawIterator &o)
    : pointer(o.pointer), myparent(o.myparent), rlw(o.rlw) {}

    bool hasNext() const { return pointer < myparent->size(); }

    BufferedRunningLengthWord<uword> &next() {
        rlw.read((*myparent)[pointer]);
        pointer = static_cast<size_t>(pointer + rlw.getNumberOfLiteralWords() + 1);
        return rlw;
    }

    const uword *dirtyWords() const {
        return myparent->data() +
        static_cast<size_t>(pointer - rlw.getNumberOfLiteralWords());
    }

    HybridBitmapRawIterator &operator=(const HybridBitmapRawIterator &other) {
        pointer = other.pointer;
        myparent = other.myparent;
        rlw = other.rlw;
        return *this;
    }

    size_t pointer;
    const std::vector<uword> *myparent;
    BufferedRunningLengthWord<uword> rlw;

    HybridBitmapRawIterator();
};
/*
###################################################################################################################
 */


template <class uword>
HybridBitmapIterator<uword>::HybridBitmapIterator(
                                                  const std::vector<uword> &parent)
: pointer(0), myparent(parent), compressedwords(0), literalwords(0), rl(0),
lw(0), b(0) {
    if (pointer < myparent.size())
        readNewRunningLengthWord();
}

template <class uword>
void HybridBitmapIterator<uword>::readNewRunningLengthWord() {
    literalwords = 0;
    compressedwords = 0;
    ConstRunningLengthWord<uword> rlw(myparent[pointer]);
    rl = rlw.getRunningLength();
    lw = rlw.getNumberOfLiteralWords();
    b = rlw.getRunningBit();
    if ((rl == 0) && (lw == 0)) {
        if (pointer < myparent.size() - 1) {
            ++pointer;
            readNewRunningLengthWord();
        } else {
            pointer = myparent.size();
        }
    }
}

template <class uword>
void fast_logicalor_tocontainer(size_t n, const HybridBitmap<uword> **inputs,
                                HybridBitmap<uword> &container) {
    class HybridBitmapPtr {

    public:
        HybridBitmapPtr(const HybridBitmap<uword> *p, bool o) : ptr(p), own(o) {}
        const HybridBitmap<uword> *ptr;
        bool own; // whether to clean

        bool operator<(const HybridBitmapPtr &o) const {
            return o.ptr->sizeInBytes() < ptr->sizeInBytes(); // backward on purpose
        }
    };

    if (n == 0) {
        container.reset();
        return;
    }
    if (n == 1) {
        container = *inputs[0];
        return;
    }
    std::priority_queue<HybridBitmapPtr> pq;
    for (size_t i = 0; i < n; i++) {
        // could use emplace
        pq.push(HybridBitmapPtr(inputs[i], false));
    }
    while (pq.size() > 2) {

        HybridBitmapPtr x1 = pq.top();
        pq.pop();

        HybridBitmapPtr x2 = pq.top();
        pq.pop();

        HybridBitmap<uword> *buffer = new HybridBitmap<uword>();
        x1.ptr->logicalor(*x2.ptr, *buffer);

        if (x1.own) {
            delete x1.ptr;
        }
        if (x2.own) {
            delete x2.ptr;
        }
        pq.push(HybridBitmapPtr(buffer, true));
    }
    HybridBitmapPtr x1 = pq.top();
    pq.pop();

    HybridBitmapPtr x2 = pq.top();
    pq.pop();

    x1.ptr->logicalor(*x2.ptr, container);

    if (x1.own) {
        delete x1.ptr;
    }
    if (x2.own) {
        delete x2.ptr;
    }
}


template <class uword> bool HybridBitmap<uword>::set(size_t i) {
    
    if (i < sizeinbits){
        return false;
    }
    if(!verbatim){
        const size_t dist = (i + wordinbits) / wordinbits - (sizeinbits + wordinbits - 1) / wordinbits;
        density = ((density*sizeinbits)+1)/(i+1);
        sizeinbits = i + 1;
        
        if (dist > 0) { // easy
            if (dist > 1)
                fastaddStreamOfEmptyWords(false, dist - 1);
            addLiteralWord(static_cast<uword>(static_cast<uword>(1) << (i % wordinbits)));
            return true;
        }
        RunningLengthWord<uword> lastRunningLengthWord(buffer[lastRLW]);
        if (lastRunningLengthWord.getNumberOfLiteralWords() == 0) {
            lastRunningLengthWord.setRunningLength(static_cast<uword>(lastRunningLengthWord.getRunningLength() - 1));
            addLiteralWord(static_cast<uword>(static_cast<uword>(1) << (i % wordinbits)));
            return true;
        }
        buffer[buffer.size() - 1] |= static_cast<uword>(static_cast<uword>(1) << (i % wordinbits));
        // check if we just completed a stream of 1s
        if (buffer[buffer.size() - 1] == static_cast<uword>(~0)) {
            // we remove the last dirty word
            buffer[buffer.size() - 1] = 0;
            buffer.resize(buffer.size() - 1);
            lastRunningLengthWord.setNumberOfLiteralWords(static_cast<uword>(lastRunningLengthWord.getNumberOfLiteralWords() - 1));
            // next we add one clean word
            addEmptyWord(true);
        }
        return true;
    }else{
        
        
        
    }
    
      return true;
}


//template <class uword> bool HybridBitmap<uword>::set(size_t i) {
//    if (i < sizeinbits)
//        return false;
//    const size_t dist = (i + wordinbits) / wordinbits -
//    (sizeinbits + wordinbits - 1) / wordinbits;
//    sizeinbits = i + 1;
//    if (dist > 0) { // easy
//        if (dist > 1)
//            fastaddStreamOfEmptyWords(false, dist - 1);
//        addLiteralWord(
//                       static_cast<uword>(static_cast<uword>(1) << (i % wordinbits)));
//        return true;
//    }
//    RunningLengthWord<uword> lastRunningLengthWord(buffer[lastRLW]);
//    if (lastRunningLengthWord.getNumberOfLiteralWords() == 0) {
//        lastRunningLengthWord.setRunningLength(
//                                               static_cast<uword>(lastRunningLengthWord.getRunningLength() - 1));
//        addLiteralWord(
//                       static_cast<uword>(static_cast<uword>(1) << (i % wordinbits)));
//        return true;
//    }
//    buffer[buffer.size() - 1] |=
//    static_cast<uword>(static_cast<uword>(1) << (i % wordinbits));
//    // check if we just completed a stream of 1s
//    if (buffer[buffer.size() - 1] == static_cast<uword>(~0)) {
//        // we remove the last dirty word
//        buffer[buffer.size() - 1] = 0;
//        buffer.resize(buffer.size() - 1);
//        lastRunningLengthWord.setNumberOfLiteralWords(static_cast<uword>(
//                                                                         lastRunningLengthWord.getNumberOfLiteralWords() - 1));
//        // next we add one clean word
//        addEmptyWord(true);
//    }
//    return true;
//}
template <class uword> void HybridBitmap<uword>::inplace_logicalnot() {
    size_t pointer(0), lastrlw(0);
    while (pointer < buffer.size()) {
        RunningLengthWord<uword> rlw(buffer[pointer]);
        lastrlw = pointer; // we save this up
        if (rlw.getRunningBit())
            rlw.setRunningBit(false);
        else
            rlw.setRunningBit(true);
        ++pointer;
        for (size_t k = 0; k < rlw.getNumberOfLiteralWords(); ++k) {
            buffer[pointer] = static_cast<uword>(~buffer[pointer]);
            ++pointer;
        }
    }
    if (sizeinbits % wordinbits != 0) {
        RunningLengthWord<uword> rlw(buffer[lastrlw]);
        const uword maskbogus =
                (static_cast<uword>(1) << (sizeinbits % wordinbits)) - 1;
        if (rlw.getNumberOfLiteralWords() > 0) { // easy case
            buffer[lastrlw + 1 + rlw.getNumberOfLiteralWords() - 1] &= maskbogus;
        } else {
            rlw.setRunningLength(rlw.getRunningLength() - 1);
            addLiteralWord(maskbogus);
        }
    }
}

template <class uword> size_t HybridBitmap<uword>::numberOfOnes() const {
    size_t tot(0);
    size_t pointer(0);
    if(verbatim){
        for (int i = 0; i < bufferSize(); i++) {
            tot += countOnes((uword)buffer[i]);
        }
    }else{
        while (pointer < buffer.size()) {
            ConstRunningLengthWord<uword> rlw(buffer[pointer]);
            if (rlw.getRunningBit()) {
                tot += static_cast<size_t>(rlw.getRunningLength() * wordinbits);
            }
            ++pointer;
            for (size_t k = 0; k < rlw.getNumberOfLiteralWords(); ++k) {
                tot += countOnes((uword)buffer[pointer]);
                ++pointer;
            }
        }
    }
    return tot;
}

template <class uword>
std::vector<size_t> HybridBitmap<uword>::toArray() const {
    std::vector<size_t> ans;
    size_t pos(0);
    size_t pointer(0);
    const size_t buffersize = buffer.size();
    while (pointer < buffersize) {
        ConstRunningLengthWord<uword> rlw(buffer[pointer]);
        const size_t productofrl =
                static_cast<size_t>(rlw.getRunningLength() * wordinbits);
        if (rlw.getRunningBit()) {
            size_t upper_limit = pos + productofrl;
            for (; pos < upper_limit; ++pos) {
                ans.push_back(pos);
            }
        } else {
            pos += productofrl;
        }
        ++pointer;
        const size_t rlwlw = rlw.getNumberOfLiteralWords();
        for (size_t k = 0; k < rlwlw; ++k) {
            uword myword = buffer[pointer];
            while (myword != 0) {
                uint64_t t = myword & (~myword + 1);
                uint32_t r = numberOfTrailingZeros(t);
                ans.push_back(pos + r);
                myword ^= t;
            }
            pos += wordinbits;
            ++pointer;
        }
    }
    return ans;
}

template <class uword>
void HybridBitmap<uword>::logicalnot(HybridBitmap &x) const {
    x.reset();
    x.density=1-density;
    if (verbatim){
        x.verbatim = true;
        for(int i=0; i < bufferSize(); i++){
            x.buffer.push_back(~buffer[i]);
        }
        x.sizeinbits = this->sizeinbits;
        return;
    }
    x.buffer.push_back(0);
    x.buffer.reserve(buffer.size());
    HybridBitmapRawIterator<uword> i = this->raw_iterator();
    if (!i.hasNext())
        return; // nothing to do
    while (true) {
        BufferedRunningLengthWord<uword> &rlw = i.next();
        if (i.hasNext()) {
            if (rlw.getRunningLength() > 0)
                x.fastaddStreamOfEmptyWords(!rlw.getRunningBit(),
                                            rlw.getRunningLength());
            if (rlw.getNumberOfLiteralWords() > 0) {
                const uword *dw = i.dirtyWords();
                for (size_t k = 0; k < rlw.getNumberOfLiteralWords(); ++k) {
                    x.addLiteralWord(~dw[k]);
                }
            }
        } else {
            if (rlw.getNumberOfLiteralWords() == 0) {
                if ((this->sizeinbits % wordinbits != 0) && !rlw.getRunningBit()) {
                    if (rlw.getRunningLength() > 1)
                        x.fastaddStreamOfEmptyWords(!rlw.getRunningBit(),
                                                    rlw.getRunningLength() - 1);
                    const uword maskbogus =
                            (static_cast<uword>(1) << (this->sizeinbits % wordinbits)) - 1;
                    x.addLiteralWord(maskbogus);
                    break;
                } else {
                    if (rlw.getRunningLength() > 0)
                        x.fastaddStreamOfEmptyWords(!rlw.getRunningBit(),
                                                    rlw.getRunningLength());
                    break;
                }
            }
            if (rlw.getRunningLength() > 0)
                x.fastaddStreamOfEmptyWords(!rlw.getRunningBit(),
                                            rlw.getRunningLength());
            const uword *dw = i.dirtyWords();
            for (size_t k = 0; k + 1 < rlw.getNumberOfLiteralWords(); ++k) {
                x.addLiteralWord(~dw[k]);
            }
            const uword maskbogus =
                    (this->sizeinbits % wordinbits != 0)
                    ? (static_cast<uword>(1) << (this->sizeinbits % wordinbits)) - 1
                    : ~static_cast<uword>(0);
            x.addLiteralWord((~dw[rlw.getNumberOfLiteralWords() - 1]) & maskbogus);
            break;
        }
    }
    x.sizeinbits = this->sizeinbits;
}


template <class uword>
size_t HybridBitmap<uword>::addLiteralWord(const uword newdata) {
    
    RunningLengthWord<uword> lastRunningLengthWord(buffer[lastRLW]);
    uword numbersofar = lastRunningLengthWord.getNumberOfLiteralWords();
    if (numbersofar >=
        RunningLengthWord<uword>::largestliteralcount) { // 0x7FFF) {
        lastRLW = buffer.size() - 1;
        RunningLengthWord<uword> lastRunningLengthWord2(buffer[lastRLW]);
        lastRunningLengthWord2.setNumberOfLiteralWords(1);
        buffer.push_back(newdata);
        return 2;
    }
    lastRunningLengthWord.setNumberOfLiteralWords(
                                                  static_cast<uword>(numbersofar + 1));
    buffer.push_back(newdata);
    return 1;
}


template <class uword>
size_t HybridBitmap<uword>::addWord(const uword newdata,
                                    const uword bitsthatmatter) {
    sizeinbits += bitsthatmatter;
    if (newdata == 0) {
        return addEmptyWord(0);
    } else if (newdata == static_cast<uword>(~0)) {
        return addEmptyWord(1);
    } else {
        return addLiteralWord(newdata);
    }
}

template <class uword>
void HybridBitmap<uword>::addVerbatim(const uword newdata) {
    addVerbatim(newdata, wordinbits);
}


template <class uword>
void HybridBitmap<uword>::addVerbatim(const uword newdata,
                                             const uword bitsthatmatter) {
    
        buffer.push_back(newdata);
        sizeinbits+=bitsthatmatter;


}

template <class uword>
inline void HybridBitmap<uword>::writeBuffer(std::ostream &out) const {
    if (!buffer.empty())
        out.write(reinterpret_cast<const char *>(&buffer[0]),
                  sizeof(uword) * buffer.size());
}

template <class uword>
inline void HybridBitmap<uword>::readBuffer(std::istream &in,
                                            const size_t buffersize) {
    buffer.resize(buffersize);
    if (buffersize > 0)
        in.read(reinterpret_cast<char *>(&buffer[0]), sizeof(uword) * buffersize);
}

template <class uword>
size_t HybridBitmap<uword>::write(std::ostream &out,
                                  const bool savesizeinbits) const {
    size_t written = 0;
    if (savesizeinbits) {
        out.write(reinterpret_cast<const char *>(&sizeinbits), sizeof(sizeinbits));
        written += sizeof(sizeinbits);
    }
    const size_t buffersize = buffer.size();
    out.write(reinterpret_cast<const char *>(&buffersize), sizeof(buffersize));
    written += sizeof(buffersize);

    if (buffersize > 0) {
        out.write(reinterpret_cast<const char *>(&buffer[0]),
                  static_cast<std::streamsize>(sizeof(uword) * buffersize));
        written += sizeof(uword) * buffersize;
    }
    return written;
}

template <class uword>
size_t HybridBitmap<uword>::write(char * out, size_t capacity,
                                  const bool savesizeinbits) const {
    size_t written = 0;
    if (savesizeinbits) {
        if(capacity < sizeof(sizeinbits)) return 0;
        capacity -= sizeof(sizeinbits);
        memcpy(out, &sizeinbits, sizeof(sizeinbits));
        out += sizeof(sizeinbits);
        written += sizeof(sizeinbits);
    }
    const size_t buffersize = buffer.size();
    if(capacity < sizeof(buffersize)) return 0;
    capacity -= sizeof(buffersize);
    memcpy(out, &buffersize, sizeof(buffersize));
    out += sizeof(buffersize);
    written += sizeof(buffersize);

    if (buffersize > 0) {
        if(capacity < sizeof(uword) * buffersize) return 0;
        memcpy(out, &buffer[0], sizeof(uword) * buffersize);
        written += sizeof(uword) * buffersize;
    }
    return written;
}


template <class uword>
size_t HybridBitmap<uword>::read(std::istream &in, const bool savesizeinbits) {
    size_t read = 0;
    if (savesizeinbits) {
        in.read(reinterpret_cast<char *>(&sizeinbits), sizeof(sizeinbits));
        read += sizeof(sizeinbits);
    } else {
        sizeinbits = 0;
    }
    size_t buffersize(0);
    in.read(reinterpret_cast<char *>(&buffersize), sizeof(buffersize));
    read += sizeof(buffersize);
    buffer.resize(buffersize);
    if (buffersize > 0) {
        in.read(reinterpret_cast<char *>(&buffer[0]),
                static_cast<std::streamsize>(sizeof(uword) * buffersize));
        read += sizeof(uword) * buffersize;
    }
    return read;
}


template <class uword>
size_t HybridBitmap<uword>::read(const char * in, size_t capacity, const bool savesizeinbits) {
    size_t read = 0;
    if (savesizeinbits) {
        if(capacity < sizeof(sizeinbits)) return 0;
        capacity -= sizeof(sizeinbits);
        memcpy(reinterpret_cast<char *>(&sizeinbits), in, sizeof(sizeinbits));
        read += sizeof(sizeinbits);
        in += sizeof(sizeinbits);
    } else {
        sizeinbits = 0;
    }
    size_t buffersize(0);
    if(capacity < sizeof(buffersize)) return 0;
    capacity -= sizeof(buffersize);
    memcpy(reinterpret_cast<char *>(&buffersize), in, sizeof(buffersize));
    in += sizeof(buffersize);
    read += sizeof(buffersize);

    buffer.resize(buffersize);
    if (buffersize > 0) {
        if(capacity < sizeof(uword) * buffersize) return 0;
        memcpy(&buffer[0], in, sizeof(uword) * buffersize);
        read += sizeof(uword) * buffersize;
    }
    return read;
}


template <class uword>
size_t HybridBitmap<uword>::padWithZeroes(const size_t totalbits) {
    size_t wordsadded = 0;
    if (totalbits <= sizeinbits)
        return wordsadded;

    size_t missingbits = totalbits - sizeinbits;

    RunningLengthWord<uword> rlw(buffer[lastRLW]);
    if (rlw.getNumberOfLiteralWords() > 0) {
        // Consume trailing zeroes of trailing literal word (past sizeinbits)
        size_t remain = sizeinbits % wordinbits;
        if (remain > 0) // Is last word partial?
        {
            size_t avail = wordinbits - remain;
            if (avail > 0) {
                if (missingbits > avail) {
                    missingbits -= avail;
                } else {
                    missingbits = 0;
                }
                sizeinbits += avail;
            }
        }
    }

    if (missingbits > 0) {
        size_t wordstoadd = missingbits / wordinbits;
        if ((missingbits % wordinbits) != 0)
            ++wordstoadd;

        wordsadded = addStreamOfEmptyWords(false, wordstoadd);
    }
    sizeinbits = totalbits;
    return wordsadded;
}

template <class uword>
HybridBitmapIterator<uword> HybridBitmap<uword>::uncompress() const {
    return HybridBitmapIterator<uword>(buffer);
}

template <class uword>
HybridBitmapRawIterator<uword> HybridBitmap<uword>::raw_iterator() const {
    return HybridBitmapRawIterator<uword>(*this);
}

template <class uword>
bool HybridBitmap<uword>::operator==(const HybridBitmap &x) const {
    HybridBitmapRawIterator<uword> i = x.raw_iterator();
    HybridBitmapRawIterator<uword> j = raw_iterator();
    if (!(i.hasNext() and j.hasNext())) { // hopefully this never happens...
        return (i.hasNext() == false) && (j.hasNext() == false);
    }
    // at this point, this should be safe:
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    BufferedRunningLengthWord<uword> &rlwj = j.next();

    while ((rlwi.size() > 0) && (rlwj.size() > 0)) {
        while ((rlwi.getRunningLength() > 0) || (rlwj.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwj.getRunningLength();
            BufferedRunningLengthWord<uword> &prey = i_is_prey ? rlwi : rlwj;
            BufferedRunningLengthWord<uword> &predator = i_is_prey ? rlwj : rlwi;
            size_t index = 0;
            const bool nonzero =
                    ((!predator.getRunningBit())
                     ? prey.nonzero_discharge(predator.getRunningLength(), index)
                     : prey.nonzero_dischargeNegated(predator.getRunningLength(),
                                                     index));
            if (nonzero) {
                return false;
            }
            if (predator.getRunningLength() - index > 0) {
                if (predator.getRunningBit()) {
                    return false;
                }
            }
            predator.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwj.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k)
                if ((rlwi.getLiteralWordAt(k) ^ rlwj.getLiteralWordAt(k)) != 0)
                    return false;
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwj.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    const bool i_remains = rlwi.size() > 0;
    BufferedRunningLengthWord<uword> &remaining = i_remains ? rlwi : rlwj;
    return !remaining.nonzero_discharge();
}

template <class uword> void HybridBitmap<uword>::swap(HybridBitmap &x) {
    buffer.swap(x.buffer);
    size_t tmp = x.sizeinbits;
    x.sizeinbits = sizeinbits;
    sizeinbits = tmp;
    tmp = x.lastRLW;
    x.lastRLW = lastRLW;
    lastRLW = tmp;
}

template <class uword>
void HybridBitmap<uword>::append(const HybridBitmap &x) {
    if (sizeinbits % wordinbits == 0) {
        // hoping for the best?
        sizeinbits += x.sizeinbits;
        ConstRunningLengthWord<uword> lRLW(buffer[lastRLW]);
        if ((lRLW.getRunningLength() == 0) &&
            (lRLW.getNumberOfLiteralWords() == 0)) {
            // it could be that the running length word is empty, in such a case,
            // we want to get rid of it!
            lastRLW = x.lastRLW + buffer.size() - 1;
            buffer.resize(buffer.size() - 1);
            buffer.insert(buffer.end(), x.buffer.begin(), x.buffer.end());
        } else {
            lastRLW = x.lastRLW + buffer.size();
            buffer.insert(buffer.end(), x.buffer.begin(), x.buffer.end());
        }
    } else {
        std::stringstream ss;
        ss << "This should really not happen! You are trying to append to a bitmap "
              "having a fractional number of words, that is,  "
           << static_cast<int>(sizeinbits) << " bits with a word size in bits of "
           << static_cast<int>(wordinbits) << ". ";
        ss << "Size of the bitmap being appended: " << x.sizeinbits << " bits."
           << std::endl;
        throw std::invalid_argument(ss.str());
    }
}

template <class uword>
BoolArray<uword> HybridBitmap<uword>::toBoolArray() const {
    BoolArray<uword> ans(sizeinbits);
    HybridBitmapIterator<uword> i = uncompress();
    size_t counter = 0;
    while (i.hasNext()) {
        ans.setWord(counter++, i.next());
    }
    return ans;
}

template <class uword>
template <class container>
void HybridBitmap<uword>::appendRowIDs(container &out,
                                       const size_t offset) const {
    size_t pointer(0);
    size_t currentoffset(offset);
    if (RESERVEMEMORY)
        out.reserve(buffer.size() + 64); // trading memory for speed.
    const size_t buffersize = buffer.size();
    while (pointer < buffersize) {
        ConstRunningLengthWord<uword> rlw(buffer[pointer]);
        const size_t productofrl =
                static_cast<size_t>(rlw.getRunningLength() * wordinbits);
        if (rlw.getRunningBit()) {
            const size_t upper_limit = currentoffset + productofrl;
            for (; currentoffset < upper_limit; ++currentoffset) {
                out.push_back(currentoffset);
            }
        } else {
            currentoffset += productofrl;
        }
        ++pointer;
        const size_t rlwlw = rlw.getNumberOfLiteralWords();
        for (uword k = 0; k < rlwlw; ++k) {
            uword currentword = buffer[pointer];
            while (currentword != 0) {
                uint64_t t = currentword & -currentword;
                uint32_t r = numberOfTrailingZeros(t);
                out.push_back(currentoffset + r);
                currentword ^= t;
            }
            currentoffset += wordinbits;
            ++pointer;
        }
    }
}

template <class uword>
bool HybridBitmap<uword>::operator!=(const HybridBitmap<uword> &x) const {
    return !(*this == x);
}

template <class uword>
bool HybridBitmap<uword>::operator==(const BoolArray<uword> &x) const {
    // could be more efficient
    return (this->toBoolArray() == x);
}

template <class uword>
bool HybridBitmap<uword>::operator!=(const BoolArray<uword> &x) const {
    // could be more efficient
    return (this->toBoolArray() != x);
}

template <class uword>
size_t HybridBitmap<uword>::addStreamOfEmptyWords(const bool v,
                                                  size_t number) {
    if (number == 0)
        return 0;
    sizeinbits += number * wordinbits;
    size_t wordsadded = 0;
    if ((RunningLengthWord<uword>::getRunningBit(buffer[lastRLW]) != v) &&
        (RunningLengthWord<uword>::size(buffer[lastRLW]) == 0)) {
        RunningLengthWord<uword>::setRunningBit(buffer[lastRLW], v);
    } else if ((RunningLengthWord<uword>::getNumberOfLiteralWords(
            buffer[lastRLW]) != 0) ||
               (RunningLengthWord<uword>::getRunningBit(buffer[lastRLW]) != v)) {
        buffer.push_back(0);
        ++wordsadded;
        lastRLW = buffer.size() - 1;
        if (v)
            RunningLengthWord<uword>::setRunningBit(buffer[lastRLW], v);
    }
    const uword runlen =
            RunningLengthWord<uword>::getRunningLength(buffer[lastRLW]);

    const uword whatwecanadd =
            number < static_cast<size_t>(
                    RunningLengthWord<uword>::largestrunninglengthcount - runlen)
            ? static_cast<uword>(number)
            : static_cast<uword>(
                    RunningLengthWord<uword>::largestrunninglengthcount - runlen);
    RunningLengthWord<uword>::setRunningLength(
            buffer[lastRLW], static_cast<uword>(runlen + whatwecanadd));

    number -= static_cast<size_t>(whatwecanadd);
    while (number >= RunningLengthWord<uword>::largestrunninglengthcount) {
        buffer.push_back(0);
        ++wordsadded;
        lastRLW = buffer.size() - 1;
        if (v)
            RunningLengthWord<uword>::setRunningBit(buffer[lastRLW], v);
        RunningLengthWord<uword>::setRunningLength(
                buffer[lastRLW], RunningLengthWord<uword>::largestrunninglengthcount);
        number -= static_cast<size_t>(
                RunningLengthWord<uword>::largestrunninglengthcount);
    }
    if (number > 0) {
        buffer.push_back(0);
        ++wordsadded;
        lastRLW = buffer.size() - 1;
        if (v)
            RunningLengthWord<uword>::setRunningBit(buffer[lastRLW], v);
        RunningLengthWord<uword>::setRunningLength(buffer[lastRLW],
                                                   static_cast<uword>(number));
    }
    return wordsadded;
}

template <class uword>
void HybridBitmap<uword>::fastaddStreamOfEmptyWords(const bool v,
                                                    size_t number) {
    if (number == 0)
        return;
    if ((RunningLengthWord<uword>::getRunningBit(buffer[lastRLW]) != v) &&
        (RunningLengthWord<uword>::size(buffer[lastRLW]) == 0)) {
        RunningLengthWord<uword>::setRunningBit(buffer[lastRLW], v);
    } else if ((RunningLengthWord<uword>::getNumberOfLiteralWords(
            buffer[lastRLW]) != 0) ||
               (RunningLengthWord<uword>::getRunningBit(buffer[lastRLW]) != v)) {
        buffer.push_back(0);
        lastRLW = buffer.size() - 1;
        if (v)
            RunningLengthWord<uword>::setRunningBit(buffer[lastRLW], v);
    }
    const uword runlen =
            RunningLengthWord<uword>::getRunningLength(buffer[lastRLW]);

    const uword whatwecanadd =
            number < static_cast<size_t>(
                    RunningLengthWord<uword>::largestrunninglengthcount - runlen)
            ? static_cast<uword>(number)
            : static_cast<uword>(
                    RunningLengthWord<uword>::largestrunninglengthcount - runlen);
    RunningLengthWord<uword>::setRunningLength(
            buffer[lastRLW], static_cast<uword>(runlen + whatwecanadd));

    number -= static_cast<size_t>(whatwecanadd);
    while (number >= RunningLengthWord<uword>::largestrunninglengthcount) {
        buffer.push_back(0);
        lastRLW = buffer.size() - 1;
        if (v)
            RunningLengthWord<uword>::setRunningBit(buffer[lastRLW], v);
        RunningLengthWord<uword>::setRunningLength(
                buffer[lastRLW], RunningLengthWord<uword>::largestrunninglengthcount);
        number -= static_cast<size_t>(
                RunningLengthWord<uword>::largestrunninglengthcount);
    }
    if (number > 0) {
        buffer.push_back(0);
        lastRLW = buffer.size() - 1;
        if (v)
            RunningLengthWord<uword>::setRunningBit(buffer[lastRLW], v);
        RunningLengthWord<uword>::setRunningLength(buffer[lastRLW],
                                                   static_cast<uword>(number));
    }
}

template <class uword>
size_t HybridBitmap<uword>::addStreamOfDirtyWords(const uword *v,
                                                  const size_t number) {
    if (number == 0)
        return 0;
    uword rlw = buffer[lastRLW];
    size_t NumberOfLiteralWords =
            RunningLengthWord<uword>::getNumberOfLiteralWords(rlw);
    if (NumberOfLiteralWords + number <=
        RunningLengthWord<uword>::largestliteralcount) {
        RunningLengthWord<uword>::setNumberOfLiteralWords(
                rlw, NumberOfLiteralWords + number);
        buffer[lastRLW] = rlw;
        sizeinbits += number * wordinbits;
        buffer.insert(buffer.end(), v, v + number);
        return number;
    }
    // we proceed the long way
    size_t howmanywecanadd =
            RunningLengthWord<uword>::largestliteralcount - NumberOfLiteralWords;
    RunningLengthWord<uword>::setNumberOfLiteralWords(
            rlw, RunningLengthWord<uword>::largestliteralcount);
    buffer[lastRLW] = rlw;
    buffer.insert(buffer.end(), v, v + howmanywecanadd);
    size_t wordadded = howmanywecanadd;
    sizeinbits += howmanywecanadd * wordinbits;
    buffer.push_back(0);
    lastRLW = buffer.size() - 1;
    ++wordadded;
    wordadded +=
            addStreamOfDirtyWords(v + howmanywecanadd, number - howmanywecanadd);
    return wordadded;
}

template <class uword>
void HybridBitmap<uword>::fastaddStreamOfDirtyWords(const uword *v,
                                                    const size_t number) {
    if (number == 0)
        return;
    uword rlw = buffer[lastRLW];
    size_t NumberOfLiteralWords =
            RunningLengthWord<uword>::getNumberOfLiteralWords(rlw);
    if (NumberOfLiteralWords + number <=
        RunningLengthWord<uword>::largestliteralcount) {
        RunningLengthWord<uword>::setNumberOfLiteralWords(
                rlw, NumberOfLiteralWords + number);
        buffer[lastRLW] = rlw;
        for (size_t i = 0; i < number; ++i)
            buffer.push_back(v[i]);
        // buffer.insert(buffer.end(), v, v+number); // seems slower than push_back?
        return;
    }
    // we proceed the long way
    size_t howmanywecanadd =
            RunningLengthWord<uword>::largestliteralcount - NumberOfLiteralWords;
    RunningLengthWord<uword>::setNumberOfLiteralWords(
            rlw, RunningLengthWord<uword>::largestliteralcount);
    buffer[lastRLW] = rlw;
    for (size_t i = 0; i < howmanywecanadd; ++i)
        buffer.push_back(v[i]);
    // buffer.insert(buffer.end(), v, v+howmanywecanadd);// seems slower than
    // push_back?
    buffer.push_back(0);
    lastRLW = buffer.size() - 1;
    fastaddStreamOfDirtyWords(v + howmanywecanadd, number - howmanywecanadd);
}

template <class uword>
size_t HybridBitmap<uword>::addStreamOfNegatedDirtyWords(const uword *v,
                                                         const size_t number) {
    if (number == 0)
        return 0;
    uword rlw = buffer[lastRLW];
    size_t NumberOfLiteralWords =
            RunningLengthWord<uword>::getNumberOfLiteralWords(rlw);
    if (NumberOfLiteralWords + number <=
        RunningLengthWord<uword>::largestliteralcount) {
        RunningLengthWord<uword>::setNumberOfLiteralWords(
                rlw, NumberOfLiteralWords + number);
        buffer[lastRLW] = rlw;
        sizeinbits += number * wordinbits;
        for (size_t k = 0; k < number; ++k)
            buffer.push_back(~v[k]);
        return number;
    }
    // we proceed the long way
    size_t howmanywecanadd =
            RunningLengthWord<uword>::largestliteralcount - NumberOfLiteralWords;
    RunningLengthWord<uword>::setNumberOfLiteralWords(
            rlw, RunningLengthWord<uword>::largestliteralcount);
    buffer[lastRLW] = rlw;
    for (size_t k = 0; k < howmanywecanadd; ++k)
        buffer.push_back(~v[k]);
    size_t wordadded = howmanywecanadd;
    sizeinbits += howmanywecanadd * wordinbits;
    buffer.push_back(0);
    lastRLW = buffer.size() - 1;
    ++wordadded;
    wordadded +=
            addStreamOfDirtyWords(v + howmanywecanadd, number - howmanywecanadd);
    return wordadded;
}
template <class uword>size_t HybridBitmap<uword>::addEmptyWord(const bool v) {
    RunningLengthWord<uword> lastRunningLengthWord(buffer[lastRLW]);
    const bool noliteralword =
            (lastRunningLengthWord.getNumberOfLiteralWords() == 0);
    // first, if the last running length word is empty, we align it
    // this
    uword runlen = lastRunningLengthWord.getRunningLength();
    if ((noliteralword) && (runlen == 0)) {
        lastRunningLengthWord.setRunningBit(v);
    }
    if ((noliteralword) && (lastRunningLengthWord.getRunningBit() == v) &&
        (runlen < RunningLengthWord<uword>::largestrunninglengthcount)) {
        lastRunningLengthWord.setRunningLength(static_cast<uword>(runlen + 1));
        return 0;
    } else {
        // we have to start anew
        buffer.push_back(0);
        lastRLW = buffer.size() - 1;
        RunningLengthWord<uword> lastRunningLengthWord2(buffer[lastRLW]);
        lastRunningLengthWord2.setRunningBit(v);
        lastRunningLengthWord2.setRunningLength(1);
        return 1;
    }
}

template <class uword>
void HybridBitmap<uword>::logicalor(const HybridBitmap &a,
                                    HybridBitmap &container) const {
    container.reset();
    container.density= (density+a.density)-(density*a.density);
    container.buffer.push_back(0);
    if (RESERVEMEMORY)
        container.buffer.reserve(buffer.size() + a.buffer.size());
    HybridBitmapRawIterator<uword> i = a.raw_iterator();
    HybridBitmapRawIterator<uword> j = raw_iterator();
    if (!(i.hasNext() and j.hasNext())) { // hopefully this never happens...
        container.setSizeInBits(sizeInBits());
        return;
    }
    // at this point, this should be safe:
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    BufferedRunningLengthWord<uword> &rlwj = j.next();

    while ((rlwi.size() > 0) && (rlwj.size() > 0)) {
        while ((rlwi.getRunningLength() > 0) || (rlwj.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwj.getRunningLength();
            BufferedRunningLengthWord<uword> &prey = i_is_prey ? rlwi : rlwj;
            BufferedRunningLengthWord<uword> &predator = i_is_prey ? rlwj : rlwi;
            if (predator.getRunningBit()) {
                container.fastaddStreamOfEmptyWords(true, predator.getRunningLength());
                prey.discardFirstWordsWithReload(predator.getRunningLength());
            } else {
                const size_t index =
                        prey.discharge(container, predator.getRunningLength());
                container.fastaddStreamOfEmptyWords(false, predator.getRunningLength() -
                                                           index);
            }
            predator.discardRunningWordsWithReload();
        }

        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwj.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                container.addWord(rlwi.getLiteralWordAt(k) | rlwj.getLiteralWordAt(k));
            }
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwj.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    const bool i_remains = rlwi.size() > 0;
    BufferedRunningLengthWord<uword> &remaining = i_remains ? rlwi : rlwj;
    remaining.discharge(container);
    container.setSizeInBits(sizeInBits() > a.sizeInBits() ? sizeInBits() : a.sizeInBits());
}

template <class uword>
void HybridBitmap<uword>::logicalorDecompress(const HybridBitmap &a,
                                    HybridBitmap &container) const {
    container.reset();
    container.density= (density+a.density)-(density*a.density);
    container.verbatim =true;
    if (RESERVEMEMORY)
        container.buffer.reserve(buffer.size() + a.buffer.size());
    
    HybridBitmapRawIterator<uword> i = a.raw_iterator();
    HybridBitmapRawIterator<uword> j = raw_iterator();
    if (!(i.hasNext() and j.hasNext())) { // hopefully this never happens...
        container.setSizeInBits(sizeInBits());
        return;
    }
    // at this point, this should be safe:
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    BufferedRunningLengthWord<uword> &rlwj = j.next();
    uword actualsizeinwords = 0;

    while ((rlwi.size() > 0) && (rlwj.size() > 0)) {
        while ((rlwi.getRunningLength() > 0) || (rlwj.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwj.getRunningLength();
            BufferedRunningLengthWord<uword> &prey = i_is_prey ? rlwi : rlwj;
            BufferedRunningLengthWord<uword> &predator = i_is_prey ? rlwj : rlwi;
            if (predator.getRunningBit()) {
                //container.fastaddStreamOfEmptyWords(true, predator.getRunningLength());
                for(int i = 0; i< predator.getRunningLength(); i++){
                    container.buffer.push_back(~0L);
                }
//                std::fill(container.buffer.begin()+actualsizeinwords,
//                        container.buffer.begin()+actualsizeinwords+predator.getRunningLength(), ~0L);
//                container.buffer.size  = buffer.size() +predator.getRunningLength() ;
                actualsizeinwords+=predator.getRunningLength();
                prey.discardFirstWordsWithReload(predator.getRunningLength());
                predator.discardFirstWordsWithReload(predator.getRunningLength());
            } else {
                const size_t index =
                        prey.dischargeDecompressed(container, predator.getRunningLength());
               // container.fastaddStreamOfEmptyWords(false, predator.getRunningLength() -
               //                                            index);
                for(int i = actualsizeinwords; i< actualsizeinwords+predator.getRunningLength() - index; i++){
                    container.buffer.push_back(0);
                }
                
//                std::fill(container.buffer.begin()+actualsizeinwords,
//                          container.buffer.begin()+actualsizeinwords+predator.getRunningLength()-index, 0);
                actualsizeinwords+=predator.getRunningLength()-index;
                predator.discardFirstWordsWithReload(predator.getRunningLength());
            }
            
        }

        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwj.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                container.buffer.push_back((rlwi.getLiteralWordAt(k) | rlwj.getLiteralWordAt(k)));
                //container.addWord(rlwi.getLiteralWordAt(k) | rlwj.getLiteralWordAt(k));
                actualsizeinwords++;
            }
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwj.discardLiteralWordsWithReload(nbre_literal);
        }
    }
//    const bool i_remains = rlwi.size() > 0;
//    BufferedRunningLengthWord<uword> &remaining = i_remains ? rlwi : rlwj;
//    remaining.discharge(container);
    container.setSizeInBits(sizeInBits() > a.sizeInBits() ? sizeInBits() : a.sizeInBits());
}

template <class uword>
size_t HybridBitmap<uword>::logicalorcount(const HybridBitmap &a) const {
    size_t answer = 0;
    HybridBitmapRawIterator<uword> i = a.raw_iterator();
    HybridBitmapRawIterator<uword> j = raw_iterator();
    if (!(i.hasNext() and j.hasNext())) { // hopefully this never happens...
        return 0;
    }
    // at this point, this should be safe:
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    BufferedRunningLengthWord<uword> &rlwj = j.next();

    while ((rlwi.size() > 0) && (rlwj.size() > 0)) {
        while ((rlwi.getRunningLength() > 0) || (rlwj.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwj.getRunningLength();
            BufferedRunningLengthWord<uword> &prey = i_is_prey ? rlwi : rlwj;
            BufferedRunningLengthWord<uword> &predator = i_is_prey ? rlwj : rlwi;
            if (predator.getRunningBit()) {
                answer += predator.getRunningLength() * wordinbits;
                prey.discardFirstWordsWithReload(predator.getRunningLength());

            } else {
                // const size_t index =
                prey.dischargeCount(predator.getRunningLength(), &answer);
            }
            predator.discardRunningWordsWithReload();
        }

        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwj.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                answer += countOnes(
                        (uword)(rlwi.getLiteralWordAt(k) | rlwj.getLiteralWordAt(k)));
            }
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwj.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    const bool i_remains = rlwi.size() > 0;
    BufferedRunningLengthWord<uword> &remaining = i_remains ? rlwi : rlwj;
    answer += remaining.dischargeCount();
    return answer;
}

template <class uword>
void HybridBitmap<uword>::logicalxor(const HybridBitmap &a,
                                     HybridBitmap &container) const {
    container.reset();
    container.buffer.push_back(0);
    if (RESERVEMEMORY)
        container.buffer.reserve(buffer.size() + a.buffer.size());
    HybridBitmapRawIterator<uword> i = a.raw_iterator();
    HybridBitmapRawIterator<uword> j = raw_iterator();
    if (!(i.hasNext() and j.hasNext())) { // hopefully this never happens...
        container.setSizeInBits(sizeInBits());
        return;
    }
    // at this point, this should be safe:
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    BufferedRunningLengthWord<uword> &rlwj = j.next();
    while ((rlwi.size() > 0) && (rlwj.size() > 0)) {
        while ((rlwi.getRunningLength() > 0) || (rlwj.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwj.getRunningLength();
            BufferedRunningLengthWord<uword> &prey = i_is_prey ? rlwi : rlwj;
            BufferedRunningLengthWord<uword> &predator = i_is_prey ? rlwj : rlwi;
            const size_t index =
                    (!predator.getRunningBit())
                    ? prey.discharge(container, predator.getRunningLength())
                    : prey.dischargeNegated(container, predator.getRunningLength());
            container.fastaddStreamOfEmptyWords(predator.getRunningBit(),
                                                predator.getRunningLength() - index);
            predator.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwj.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k)
                container.addWord(rlwi.getLiteralWordAt(k) ^ rlwj.getLiteralWordAt(k));
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwj.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    const bool i_remains = rlwi.size() > 0;
    BufferedRunningLengthWord<uword> &remaining = i_remains ? rlwi : rlwj;
    remaining.discharge(container);
    container.setSizeInBits(sizeInBits() > a.sizeInBits() ? sizeInBits() : a.sizeInBits());
}

template <class uword>
size_t HybridBitmap<uword>::logicalxorcount(const HybridBitmap &a) const {
    HybridBitmapRawIterator<uword> i = a.raw_iterator();
    HybridBitmapRawIterator<uword> j = raw_iterator();
    if (!i.hasNext())
        return a.numberOfOnes();
    if (!j.hasNext())
        return this->numberOfOnes();

    size_t answer = 0;

    // at this point, this should be safe:
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    BufferedRunningLengthWord<uword> &rlwj = j.next();
    while ((rlwi.size() > 0) && (rlwj.size() > 0)) {
        while ((rlwi.getRunningLength() > 0) || (rlwj.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwj.getRunningLength();
            BufferedRunningLengthWord<uword> &prey = i_is_prey ? rlwi : rlwj;
            BufferedRunningLengthWord<uword> &predator = i_is_prey ? rlwj : rlwi;
            size_t index;

            if (predator.getRunningBit()) {
                index =
                        prey.dischargeCountNegated(predator.getRunningLength(), &answer);
            } else {
                index = prey.dischargeCount(predator.getRunningLength(), &answer);
            }
            if (predator.getRunningBit())
                answer += (predator.getRunningLength() - index) * wordinbits;

            predator.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwj.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                answer += countOnes(
                        (uword)(rlwi.getLiteralWordAt(k) ^ rlwj.getLiteralWordAt(k)));
            }
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwj.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    const bool i_remains = rlwi.size() > 0;
    BufferedRunningLengthWord<uword> &remaining = i_remains ? rlwi : rlwj;
    answer += remaining.dischargeCount();
    return answer;
}



template <class uword>
void HybridBitmap<uword>::logicaland(const HybridBitmap &a,
                                     HybridBitmap &container) const {
    container.reset();
    container.density=density*a.density;
    container.buffer.push_back(0);
    
    if (RESERVEMEMORY)
        container.buffer.reserve(buffer.size() > a.buffer.size() ? buffer.size()
                                                                 : a.buffer.size());
    HybridBitmapRawIterator<uword> i = a.raw_iterator();
    HybridBitmapRawIterator<uword> j = raw_iterator();
    if (!(i.hasNext() and j.hasNext())) { // hopefully this never happens...
        container.setSizeInBits(sizeInBits());
        return;
    }
    // at this point, this should be safe:
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    BufferedRunningLengthWord<uword> &rlwj = j.next();

    while ((rlwi.size() > 0) && (rlwj.size() > 0)) {
        while ((rlwi.getRunningLength() > 0) || (rlwj.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwj.getRunningLength();
            BufferedRunningLengthWord<uword> &prey(i_is_prey ? rlwi : rlwj);
            BufferedRunningLengthWord<uword> &predator(i_is_prey ? rlwj : rlwi);
            if (!predator.getRunningBit()) {
                container.fastaddStreamOfEmptyWords(false, predator.getRunningLength());
                prey.discardFirstWordsWithReload(predator.getRunningLength());
            } else {
                const size_t index =
                        prey.discharge(container, predator.getRunningLength());
                container.fastaddStreamOfEmptyWords(false, predator.getRunningLength() -
                                                           index);
            }
            predator.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwj.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                container.addWord(rlwi.getLiteralWordAt(k) & rlwj.getLiteralWordAt(k));
            }
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwj.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    container.setSizeInBits(sizeInBits());
    container.setSizeInBits(sizeInBits() > a.sizeInBits() ? sizeInBits() : a.sizeInBits());
}

template <class uword>
void HybridBitmap<uword>::logicalandnot(const HybridBitmap &a,
                                        HybridBitmap &container) const {
    if (RESERVEMEMORY)
        container.buffer.reserve(buffer.size() > a.buffer.size() ? buffer.size()
                                                                 : a.buffer.size());
    HybridBitmapRawIterator<uword> i = raw_iterator();
    HybridBitmapRawIterator<uword> j = a.raw_iterator();
    if (!j.hasNext()) { // the other fellow is empty
        container = *this; // just copy, stupidly, the data
        return;
    }
    if (!(i.hasNext())) { // hopefully this never happens...
        container.setSizeInBits(sizeInBits());
        return;
    }
    // at this point, this should be safe:
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    BufferedRunningLengthWord<uword> &rlwj = j.next();

    while ((rlwi.size() > 0) && (rlwj.size() > 0)) {
        while ((rlwi.getRunningLength() > 0) || (rlwj.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwj.getRunningLength();
            BufferedRunningLengthWord<uword> &prey(i_is_prey ? rlwi : rlwj);
            BufferedRunningLengthWord<uword> &predator(i_is_prey ? rlwj : rlwi);
            if (((predator.getRunningBit()) && (i_is_prey)) ||
                ((!predator.getRunningBit()) && (!i_is_prey))) {
                container.fastaddStreamOfEmptyWords(false, predator.getRunningLength());
                prey.discardFirstWordsWithReload(predator.getRunningLength());
            } else if (i_is_prey) {
                const size_t index =
                        prey.discharge(container, predator.getRunningLength());
                container.fastaddStreamOfEmptyWords(false, predator.getRunningLength() -
                                                           index);
            } else {
                const size_t index =
                        prey.dischargeNegated(container, predator.getRunningLength());
                container.fastaddStreamOfEmptyWords(true, predator.getRunningLength() -
                                                          index);
            }
            predator.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwj.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                container.addWord(rlwi.getLiteralWordAt(k) & ~rlwj.getLiteralWordAt(k));
            }
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwj.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    const bool i_remains = rlwi.size() > 0;
    if (i_remains) {
        rlwi.discharge(container);
    }
    container.setSizeInBits(sizeInBits());
}

template <class uword>
size_t HybridBitmap<uword>::logicalandnotcount(const HybridBitmap &a) const {
    HybridBitmapRawIterator<uword> i = raw_iterator();
    HybridBitmapRawIterator<uword> j = a.raw_iterator();
    if (!j.hasNext()) { // the other fellow is empty
        return this->numberOfOnes();
    }
    if (!(i.hasNext())) { // hopefully this never happens...
        return 0;
    }
    size_t answer = 0;
    // at this point, this should be safe:
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    BufferedRunningLengthWord<uword> &rlwj = j.next();

    while ((rlwi.size() > 0) && (rlwj.size() > 0)) {
        while ((rlwi.getRunningLength() > 0) || (rlwj.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwj.getRunningLength();
            BufferedRunningLengthWord<uword> &prey(i_is_prey ? rlwi : rlwj);
            BufferedRunningLengthWord<uword> &predator(i_is_prey ? rlwj : rlwi);
            if (((predator.getRunningBit()) && (i_is_prey)) ||
                ((!predator.getRunningBit()) && (!i_is_prey))) {
                prey.discardFirstWordsWithReload(predator.getRunningLength());
            } else if (i_is_prey) {
                prey.dischargeCount(predator.getRunningLength(), &answer);
            } else {
                const size_t index =
                        prey.dischargeCountNegated(predator.getRunningLength(), &answer);
                answer += (predator.getRunningLength() - index) * wordinbits;
            }
            predator.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwj.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                answer += countOnes(
                        (uword)(rlwi.getLiteralWordAt(k) & (~rlwj.getLiteralWordAt(k))));
            }
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwj.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    const bool i_remains = rlwi.size() > 0;
    if (i_remains) {
        answer += rlwi.dischargeCount();
    }
    return answer;
}

template <class uword>
size_t HybridBitmap<uword>::logicalandcount(const HybridBitmap &a) const {
    HybridBitmapRawIterator<uword> i = a.raw_iterator();
    HybridBitmapRawIterator<uword> j = raw_iterator();
    if (!(i.hasNext() and j.hasNext())) { // hopefully this never happens...
        return 0;
    }
    size_t answer = 0;
    // at this point, this should be safe:
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    BufferedRunningLengthWord<uword> &rlwj = j.next();

    while ((rlwi.size() > 0) && (rlwj.size() > 0)) {
        while ((rlwi.getRunningLength() > 0) || (rlwj.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwj.getRunningLength();
            BufferedRunningLengthWord<uword> &prey(i_is_prey ? rlwi : rlwj);
            BufferedRunningLengthWord<uword> &predator(i_is_prey ? rlwj : rlwi);
            if (!predator.getRunningBit()) {
                prey.discardFirstWordsWithReload(predator.getRunningLength());
            } else {
                // const size_t index =
                prey.dischargeCount(predator.getRunningLength(), &answer);
            }
            predator.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwj.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                answer += countOnes(
                        (uword)(rlwi.getLiteralWordAt(k) & rlwj.getLiteralWordAt(k)));
            }
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwj.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    return answer;
}

template <class uword>
bool HybridBitmap<uword>::intersects(const HybridBitmap &a) const {
    HybridBitmapRawIterator<uword> i = a.raw_iterator();
    HybridBitmapRawIterator<uword> j = raw_iterator();
    if (!(i.hasNext() and j.hasNext())) { // hopefully this never happens...
        return false;
    }
    // at this point, this should be safe:
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    BufferedRunningLengthWord<uword> &rlwj = j.next();

    while ((rlwi.size() > 0) && (rlwj.size() > 0)) {
        while ((rlwi.getRunningLength() > 0) || (rlwj.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwj.getRunningLength();
            BufferedRunningLengthWord<uword> &prey(i_is_prey ? rlwi : rlwj);
            BufferedRunningLengthWord<uword> &predator(i_is_prey ? rlwj : rlwi);
            if (!predator.getRunningBit()) {
                prey.discardFirstWordsWithReload(predator.getRunningLength());
            } else {
                size_t index = 0;
                bool isnonzero =
                        prey.nonzero_discharge(predator.getRunningLength(), index);
                if (isnonzero)
                    return true;
            }
            predator.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwj.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                if ((rlwi.getLiteralWordAt(k) & rlwj.getLiteralWordAt(k)) != 0)
                    return true;
            }
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwj.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    return false;
}

template <class uword>
BitmapStatistics HybridBitmap<uword>::computeStatistics() const {
    BitmapStatistics bs;
    HybridBitmapRawIterator<uword> i = raw_iterator();
    while (i.hasNext()) {
        BufferedRunningLengthWord<uword> &brlw(i.next());
        ++bs.runningwordmarker;
        bs.totalliteral += brlw.getNumberOfLiteralWords();
        bs.totalcompressed += brlw.getRunningLength();
        if (brlw.getRunningLength() ==
            RunningLengthWord<uword>::largestrunninglengthcount) {
            ++bs.maximumofrunningcounterreached;
        }
    }
    return bs;
};

template <class uword> void HybridBitmap<uword>::debugprintout() const {
    std::cout << "==printing out HybridBitmap==" << std::endl;
    std::cout << "Number of compressed words: " << buffer.size() << std::endl;
    size_t pointer = 0;
    while (pointer < buffer.size()) {
        ConstRunningLengthWord<uword> rlw(buffer[pointer]);
        bool b = rlw.getRunningBit();
        const uword rl = rlw.getRunningLength();
        const uword lw = rlw.getNumberOfLiteralWords();
        std::cout << "pointer = " << pointer << " running bit=" << b
                  << " running length=" << rl << " lit. words=" << lw << std::endl;
        for (uword j = 0; j < lw; ++j) {
            const uword &w = buffer[pointer + j + 1];
            std::cout << toBinaryString(w) << std::endl;
        }
        pointer += lw + 1;
    }
    std::cout << "==END==" << std::endl;
};


template <class uword>
size_t HybridBitmap<uword>::sizeOnDisk(const bool savesizeinbits) const {
    return (savesizeinbits ? sizeof(sizeinbits) : 0) + sizeof(size_t) +
           sizeof(uword) * buffer.size();
};

template <class uword>
void HybridBitmap<uword>::maj(const HybridBitmap &a,const HybridBitmap &b, HybridBitmap &container) const{
    //HybridBitmap container = new HybridBitmap(true, this.actualsizeinwords);
    container.reset();
    container.verbatim = true;
    container.buffer.reserve(bufferSize());
    double ab = density*a.density;
    double bc = a.density*b.density;
    double ac = density*b.density;
    container.density = ab+bc-ab*bc+ac-(ab+bc-ab*bc)*ac;
    for (int i = 0; i < buffer.size(); i++) {
        container.buffer.push_back( (buffer[i] & a.buffer[i]) | (a.buffer[i] & b.buffer[i])
                              | (buffer[i] & b.buffer[i]) );
    }
    //container.actualsizeinwords = this.actualsizeinwords;
    container.sizeinbits = sizeinbits;

}


template <class uword>
void HybridBitmap<uword>::majInPlace(const HybridBitmap &a,const HybridBitmap &b) {
    //HybridBitmap container = new HybridBitmap(true, this.actualsizeinwords);
   // container.reset();
    //container.verbatim = true;
    //container.buffer.reserve(bufferSize());
    double ab = density*a.density;
    double bc = a.density*b.density;
    double ac = density*b.density;
    density = ab+bc-ab*bc+ac-(ab+bc-ab*bc)*ac;
    for (int i = 0; i < buffer.size(); i++) {
        buffer[i]=( (buffer[i] & a.buffer[i]) | (a.buffer[i] & b.buffer[i])
                                    | (buffer[i] & b.buffer[i]) );
    }
    //container.actualsizeinwords = this.actualsizeinwords;
    //container.sizeinbits = sizeinbits;

}


template <class uword>
void HybridBitmap<uword>::selectMultiplicationInPlace(const HybridBitmap &a,const HybridBitmap &FS){
    //HybridBitmap container = new HybridBitmap(true, this.actualsizeinwords);
    //container.reset();
    //container.verbatim = true;
    //container.buffer.reserve(bufferSize());
    double ab = (1 - a.density)*density;
    double bc = a.density*FS.density;
    density = ab+bc-ab*bc;
    
//    *this = a.Not().And(this).Or(a.And(FS));
    
//    buffer[i]=( (~a.buffer[i] & buffer[i]) | (a.buffer[i] & FS.buffer[i]) );
    for (int i = 0; i < buffer.size(); i++) {
        buffer[i]=( (~a.buffer[i] & buffer[i]) | (a.buffer[i] & FS.buffer[i]) );
    }
    //container.actualsizeinwords = this.actualsizeinwords;
    //container.sizeinbits = sizeinbits;

}



template <class uword>
void HybridBitmap<uword>::selectMultiplication(const HybridBitmap &res,const HybridBitmap &FS, HybridBitmap &container) const{
    //HybridBitmap container = new HybridBitmap(true, this.actualsizeinwords);
    container.reset();
    container.verbatim = true;
    container.buffer.reserve(bufferSize());
    double ab = (1 - density)*res.density;
    double bc = density*FS.density;
    container.density = ab+bc-ab*bc;
    for (int i = 0; i < buffer.size(); i++) {
        container.buffer.push_back( (~buffer[i] & res.buffer[i]) | (buffer[i] & FS.buffer[i]) );
    }
    //container.actualsizeinwords = this.actualsizeinwords;
    container.sizeinbits = sizeinbits;
    
}

template <class uword>
void HybridBitmap<uword>::shiftRow(const HybridBitmap &this_bitmap, HybridBitmap &other_bitmap){
    
    for (int i = 0; i < buffer.size(); i++) {
        buffer[i] = buffer[i] ^ (this_bitmap.buffer[i] & other_bitmap.buffer[i]);
    }
}


template <class uword>
void HybridBitmap<uword>::shiftRowWithCarry(const HybridBitmap &this_bitmap, HybridBitmap &other_bitmap, HybridBitmap &carry){
    
    
    
    
}








template <class uword>
void HybridBitmap<uword>::And(const HybridBitmap &a, HybridBitmap &container) const {
    container.density=density*a.density;
    if (verbatim && a.verbatim) {
        //if(container.density<andThreshold){
        if(std::min(density, a.density)<andThreshold){
            if(container.density==0){
                container.verbatim=false;
                container.setSizeInBits(std::max(sizeinbits, a.sizeinbits),false);
            }else{
                andVerbatimCompress(a, container);
            }
        }else{
            andVerbatim(a, container);
        }
    } else if(verbatim || a.verbatim) {
        if(container.density==0){
            container.verbatim=false;
            container.setSizeInBits(std::max(sizeinbits, a.sizeinbits),false);
        }else{
            andHybridCompress(a, container);
            container.setSizeInBits(std::max(sizeinbits, a.sizeinbits));
        }
    }else{
        if(container.density==0){
            container.verbatim=false;
            container.setSizeInBits(std::max(sizeinbits, a.sizeinbits),false);
        }else{
            //container.reserve(actualsizeinwords > a.actualsizeinwords ? actualsizeinwords : a.actualsizeinwords);
            logicaland(a, container);
        }
    }
    //        container.age = Math.max(this.age, a.age)+1;
    ////        if(container.age>20){
    ////            container.density=container.cardinality();
    ////        container.age=0;
    ////        }

}

template <class uword>
void HybridBitmap<uword>::AndInPlace(const HybridBitmap &a ) {
    density = density*a.density;
    if (verbatim && a.verbatim) {
        //if(container.density<andThreshold){
        if(std::min(density, a.density)<andThreshold){
            if(density==0){
                this->reset();
                this->setSizeInBits(std::max(sizeinbits, a.sizeinbits),false);
            }else{
                andVerbatimCompressInPlace(a);
            }
        }else{
            andVerbatimInPlace(a);
        }
    } else if(verbatim || a.verbatim) {
        if(density==0){
            this->reset();
            this->setSizeInBits(std::max(sizeinbits, a.sizeinbits),false);
        }else{
            andHybridCompressInPlace(a);
            this->setSizeInBits(std::max(sizeinbits, a.sizeinbits));
        }
    }else{
        if(density==0){
            this->reset();
            this->setSizeInBits(std::max(sizeinbits, a.sizeinbits),false);
        }else{
            //container.reserve(actualsizeinwords > a.actualsizeinwords ? actualsizeinwords : a.actualsizeinwords);
            logicalandInPlace(a);
        }
    }
}



template <class uword>
void HybridBitmap<uword>::andVerbatimCompressInPlace(const HybridBitmap &a ){
    for (int i = 0; i < buffer.size(); i++) {
         buffer[i] = buffer[i] & a.buffer[i];
    }
    
}

template <class uword>
void HybridBitmap<uword>::andVerbatimInPlace(const HybridBitmap &a ){
    density = density*a.density;

    for(int i=0; i<buffer.size();i++){
        buffer[i] = buffer[i]& a.buffer[i];
    }
    //container.actualsizeinwords=actualsizeinwords;
    
}

template <class uword>
void HybridBitmap<uword>::andHybridCompressInPlace(const HybridBitmap &a ){
    int j = 0;
    int i=0;
    density=density*a.density;
    
    if (verbatim) { // this is verbatim
        ConstRunningLengthWord<uword> rlwa(a.buffer[0]);
        size_t lastrlwa = 0;
        //BufferedRunningLengthWord<uword> &rlwa = new BufferedRunningLengthWord<uword>(a.buffer, 0);
        while (i < buffer.size()) {
            if (rlwa.getRunningBit()) { // fill of ones
//                for (j = 0; j < rlwa.getRunningLength(); j++) {
//
//                    container.add(buffer[i]);
//                    //container.setbits+=Long.bitCount(newdata);
//                    i++;
//                }
                i += rlwa.getRunningLength();
            } else {
                for (j = 0; j < rlwa.getRunningLength(); j++) {

                    buffer[i] = 0;
                    //container.setbits+=Long.bitCount(newdata);
                    i++;
                }
//                container.addStreamOfEmptyWords(false, rlwa.getRunningLength());
//                i+=rlwa.getRunningLength();
            }
            
            for( j=0; j<rlwa.getNumberOfLiteralWords(); j++){
                buffer[i] = buffer[i] & a.buffer[lastrlwa + j + 1];
                //container.setbits+=Long.bitCount(newdata);
                i++;
            }
            lastrlwa +=rlwa.getNumberOfLiteralWords()+1;
            if(lastrlwa >= a.bufferSize()){
                break;
            }
            rlwa = (a.buffer[lastrlwa]);
            //rlwa.position += rlwa.getNumberOfLiteralWords() + 1;
            
        }
    } else { // a is verbatim
        ConstRunningLengthWord<uword> rlw(buffer[0]);
        size_t lastrlw = 0;
        //rlw = new RunningLengthWord(this.buffer, 0);
        
        while (i < a.buffer.size()) {
            if (rlw.getRunningBit()) { // fill of ones
                for (j = 0; j < rlw.getRunningLength(); j++) {
                    
                    buffer[i] = a.buffer[i];
                    //container.setbits+=Long.bitCount(newdata);
                    i++;
                }
            } else {
                for (j = 0; j < rlw.getRunningLength(); j++) {
                    
                    buffer[i] = 0;
                    //container.setbits+=Long.bitCount(newdata);
                    i++;
                }
//                container.addStreamOfEmptyWords(false, rlw.getRunningLength());
//                i+=rlw.getRunningLength();
            }
            
            for( j=0; j<rlw.getNumberOfLiteralWords(); j++){
                
                buffer[i] = a.buffer[i] & buffer[lastrlw + j + 1];
                //container.setbits+=Long.bitCount(newdata);
                i++;
            }
            lastrlw +=rlw.getNumberOfLiteralWords()+1;
            if(lastrlw >= bufferSize()){
                break;
            }
            rlw = (buffer[lastrlw]);
            //rlw.position += rlw.getNumberOfLiteralWords() + 1;
            
        }
    }
}

template <class uword>
void HybridBitmap<uword>::logicalandInPlace(const HybridBitmap &a ){
    logicaland(a,*this);
}


template <class uword>
void HybridBitmap<uword>::andNotVerbatimCompress(const HybridBitmap &a, HybridBitmap &container) const {
    container.buffer.reserve(bufferSize());
    container.verbatim = false;
    container.density=density*(1-a.density);
    for (int i = 0; i < bufferSize(); i++) {
        container.add(buffer[i] & ~a.buffer[i]);
    }
    
}


template <class uword>
void HybridBitmap<uword>::andNot(const HybridBitmap &a, HybridBitmap &container) const {
    container.density=density*(1-a.density);
    //container.sizeinbits=this.sizeinbits;
    if (verbatim && a.verbatim) {
        //if(container.density<andThreshold){
        if(std::min(density, (1-a.density))<andThreshold){
            if(density==0){
                container.verbatim=false;
                //container.addStreamOfEmptyWords(false, (this.sizeinbits>>>6));
                container.setSizeInBits(sizeinbits,false);
                container.density=0;
            }else{
                container.reset();
                andNotVerbatimCompress(a, container);    }
            
        }else{
            container.reset();
            andNotVerbatim(a, container);
            
        }
        
    } else if(verbatim || a.verbatim) {
        if(density==0){
            container.verbatim=false;
            //container.addStreamOfEmptyWords(false, (this.sizeinbits>>>6));
            container.setSizeInBits(sizeinbits,false);
        }else{
            andNotHybridCompress(a, container);}
        
    }else{
        if(density==0){
            container.verbatim=false;
            //container.addStreamOfEmptyWords(false, (this.sizeinbits>>>6));
            container.setSizeInBits(sizeinbits,false);
        }else{
            container.buffer.reserve(bufferSize() > a.bufferSize() ? bufferSize() : a.bufferSize());
            logicalandnot(a, container);}
    }
    //        container.age = Math.max(this.age, a.age)+1;
    //        if(container.age>20){
    //            container.density=container.cardinality();
    //        container.age=0;
    //        }
}




template <class uword>
bool HybridBitmap<uword>::compareBitmap(const HybridBitmap &a) const {
    int i =0;
    int j = 0;
    int k = 0;
    int runLength = 0;
    if(sizeinbits != a.sizeinbits){
        return false;
    }else{
        if(false){
            return false;
        }else{
            if(verbatim){
                if(a.verbatim){
                    for(int i=0; i<buffer.size();i++){
                        if(buffer[i] != (buffer[i] | a.buffer[i])){
                            return false;
                        }
                    }
                    return true;
                }else{
                    ConstRunningLengthWord<uword> rlwa(a.buffer[0]);
                    size_t lastrlwa = 0;
                    while (i < bufferSize()) {
                        runLength=(int) rlwa.getRunningLength();
                        if (rlwa.getRunningBit()) { // fill of ones
                            
                            for (j = 0; j < runLength; j++) {
                                uword temp = buffer[i];
                                uword valueOfOneByte = 255;
                                for(int k=0; k < sizeof(uword); k++){
                                    if( (temp & valueOfOneByte) != valueOfOneByte){
                                        return false;
                                    }
                                    temp = temp >> 8;
                                }
                               
                            }
                        i+=runLength;
                        } else {
                            //container.addStreamOfEmptyWords(false, a.rlw.getRunningLength());
                            //i,i+runLength
                            for (j = 0; j < runLength; j++) {
                                uword temp = buffer[i];
                                uword valueOfOneByte = 0;
                                for( k=0; k < sizeof(uword); k++){
                                    if( (temp | valueOfOneByte) != valueOfOneByte){
                                        return false;
                                    }
                                    temp = temp >> 1;
                                }
                            }
                            //return true;
                            i+=runLength;
                        }
                        for(j=0; j<rlwa.getNumberOfLiteralWords(); j++){
                            if(buffer[i] != (buffer[i] | a.buffer[lastrlwa+j+1])){
                                return false;
                            }
                            i++;
                        }
                        lastrlwa += rlwa.getNumberOfLiteralWords() + 1;
                        rlwa = (a.buffer[lastrlwa]);
                    }
                }
                return true;
            }else if (a.verbatim){
                ConstRunningLengthWord<uword> rlw(buffer[0]);
                size_t lastrlw = 0;
                while (i < a.bufferSize()) {
                    runLength=(int) rlw.getRunningLength();
                    if (rlw.getRunningBit()) { // fill of ones
                        
                        for (j = 0; j < runLength; j++) {
                            uword temp = a.buffer[i];
                            uword valueOfOneByte = 255;
                            for( k=0; k < sizeof(uword); k++){
                                if( (temp & valueOfOneByte) != valueOfOneByte){
                                    return false;
                                }
                                temp = temp >> 1;
                            }
                        }
                       // return true;
                        i+=runLength;
                    } else {
                        //container.addStreamOfEmptyWords(false, a.rlw.getRunningLength());
                        //i,i+runLength
                        for (j = 0; j < runLength; j++) {
                            uword temp = a.buffer[i];
                            uword valueOfOneByte = 0;
                            for( k=0; k < sizeof(uword); k++){
                                if( (temp | valueOfOneByte) != valueOfOneByte){
                                    return false;
                                }
                                temp = temp >> 8;
                            }
                        }
                        //return true;
                        i+=runLength;
                    }
                    for(j=0; j<rlw.getNumberOfLiteralWords(); j++){
                        uword temp = (a.buffer[i] | buffer[lastrlw+j+1]);
                        if(a.buffer[i] != temp){
                            return false;
                        }
                        i++;
                    }
                    lastrlw += rlw.getNumberOfLiteralWords() + 1;
                    rlw = (buffer[lastrlw]);
                }
                return true;
            }else{
                ConstRunningLengthWord<uword> rlw(buffer[0]);
                size_t lastrlw = 0;
                ConstRunningLengthWord<uword> rlwa(a.buffer[0]);
                size_t lastrlwa = a.lastRLW;
                runLength=(int) rlw.getRunningLength();
                int runLengtha=(int) rlwa.getRunningLength();
                if(runLength != runLengtha){
                    return false;
                }else{
                    if (rlw.getRunningBit() != rlwa.getRunningBit()) {
                        return false;
                    }
                    if(rlw.getNumberOfLiteralWords() != rlwa.getNumberOfLiteralWords()){
                        return false;
                    }else{
                        for(j=0; j<rlw.getNumberOfLiteralWords(); j++){
                            uword temp = (a.buffer[lastrlw+j+1] | buffer[lastrlw+j+1]);
                            if(a.buffer[lastrlw+j+1] != temp){
                                return false;
                            }
                        }
                        lastrlw += rlw.getNumberOfLiteralWords() + 1;
                        rlw = (buffer[lastrlw]);
                    }
                }
                return true;
            }
        }
    }
    return true;
}

template <class uword>
void HybridBitmap<uword>::andVerbatimCompress(const HybridBitmap &a, HybridBitmap &container) const{

    container.buffer.reserve(bufferSize());
    //container.verbatim = false;
    for (int i = 0; i < buffer.size(); i++) {
        container.addWord(buffer[i] & a.buffer[i]);
    }


}

template <class uword>
void HybridBitmap<uword>::Or(const HybridBitmap &a, HybridBitmap &container) const {
    //double expDens = (this.setbits+a.setbits)/(double)(this.sizeinbits)-(this.setbits/(double)this.sizeinbits*a.setbits/(double)a.sizeinbits);
    container.density= (density+a.density)-(density*a.density);
    //    container.sizeinbits=this.sizeinbits;
    if (verbatim && a.verbatim) {
        if(container.density>(1-orThreshold)){
            //if(Math.max(this.density, a.density)>){
            orVerbatimCompress(a,container);
        }else
            orVerbatim(a, container);
    }

    else if(verbatim || a.verbatim){
        if(container.density>(1-orThreshold)){
            orHybridCompress(a,container);
        }else{
            orHybrid(a, container);
        }
    }else{
        //if(container.density>orThreshold){
        if(std::max(density, a.density)>orThreshold){
            logicalorDecompress(a, container);
        }else{
            container.buffer.reserve(bufferSize() + a.bufferSize());
            logicalor(a, container);}
    }
    //        container.age = Math.max(this.age, a.age)+1;
    //        if(container.age>20){
    //            container.density=container.cardinality();
    //        container.age=0;
    //        }

}

template <class uword>
void HybridBitmap<uword>::orVerbatimCompress(const HybridBitmap &a, HybridBitmap &container) const{                                  //container.reserve(this.actualsizeinwords);
    container.verbatim = false;
    container.density= (density+a.density)-(density*a.density);
    container.buffer.reserve(bufferSize());
    
    for (int i = 0; i < buffer.size(); i++) {
        container.addWord(buffer[i] | a.buffer[i]);

    }
}

template <class uword>
void HybridBitmap<uword>::Xor(const HybridBitmap &a, HybridBitmap &container) const{
    //final HybridBitmap container = new HybridBitmap();

    //double expDens = (this.setbits+a.setbits)/(double)(this.sizeinbits)-(this.setbits*a.setbits)/(double)(this.sizeinbits*a.sizeinbits);
    //double expDens = this.setbits/(double)this.sizeinbits*(a.sizeinbits-a.setbits)/(double)a.sizeinbits+a.setbits/(double)a.sizeinbits*(this.sizeinbits-this.setbits)/(double)this.sizeinbits;
    container.density=density*(1-a.density)+a.density*(1-density);
    //    container.sizeinbits=this.sizeinbits;
    if (verbatim && a.verbatim) {
        xorVerbatim(a, container);
    }else if(verbatim || a.verbatim){
        xorHybrid(a, container);
    }else{
        //if(container.density>orThreshold){
        if(std::max(density, a.density)>orThreshold){
            logicalxorDecompress(a, container);
        }else{
            container.buffer.reserve(bufferSize() + a.bufferSize());
            logicalxor(a, container);
            
        }
    }
    //        container.age = Math.max(this.age, a.age)+1;
    //        if(container.age>20){
    //            container.density=container.cardinality();
    //        container.age=0;
    //        }

}



template <class uword>
void HybridBitmap<uword>::XorInPlace(const HybridBitmap &a ) {
    //final HybridBitmap container = new HybridBitmap();

    //double expDens = (this.setbits+a.setbits)/(double)(this.sizeinbits)-(this.setbits*a.setbits)/(double)(this.sizeinbits*a.sizeinbits);
    //double expDens = this.setbits/(double)this.sizeinbits*(a.sizeinbits-a.setbits)/(double)a.sizeinbits+a.setbits/(double)a.sizeinbits*(this.sizeinbits-this.setbits)/(double)this.sizeinbits;
    density=density*(1-a.density)+a.density*(1-density);
    //    container.sizeinbits=this.sizeinbits;
    if (verbatim && a.verbatim) {
        inPlaceXorVerbatim(a);
    }else if(verbatim || a.verbatim){
        inPlaceXorHybrid(a);
    }else{
        //if(container.density>orThreshold){
        if(std::max(density, a.density)>orThreshold){
            //inPlaceLogicalxorDecompress(a);
            std::cout<<"inPlaceLogicalxorDecompress";
        }else{
            //container.buffer.reserve(bufferSize() + a.bufferSize());
            //inPlaceLogicalxor(a);
            this->reset();
            logicalxor(a, *this);
            std::cout<<"inPlaceLogicalxor";


        }
    }
    //        container.age = Math.max(this.age, a.age)+1;
    //        if(container.age>20){
    //            container.density=container.cardinality();
    //        container.age=0;
    //        }

}

/*
template <class uword>
void HybridBitmap<uword>::Not(HybridBitmap &container) const{

    container = *this;



    container.density=1-density;
    if (verbatim){
        for(int i=0; i < buffer.size(); i++){
            container.buffer[i]=~buffer[i];
        }
    }else{
        if (!i.hasNext())
            return;
        while (true) {
            RunningLengthWord<> rlw1 = i.next();
            rlw1.setRunningBit(!rlw1.getRunningBit());
            for (int j = 0; j < rlw1.getNumberOfLiteralWords(); ++j) {
                i.buffer()[i.literalWords() + j] = ~i.buffer()[i.literalWords() + j];
            }

            if (!i.hasNext()) {// must potentially adjust the last literal word
                final int usedbitsinlast = this.sizeinbits % wordinbits;
                if (usedbitsinlast == 0)
                    return;
                if (rlw1.getNumberOfLiteralWords() == 0) {
                    if((rlw1.getRunningLength()>0) && (rlw1.getRunningBit())) {
                        rlw1.setRunningLength(rlw1.getRunningLength()-1);
                        this.addLiteralWord((~0l) >>> (wordinbits - usedbitsinlast));
                    }
                    return;
                }
                i.buffer()[i.literalWords() + rlw1.getNumberOfLiteralWords() - 1] &= ((~0l) >>> (wordinbits - usedbitsinlast));
                return;
            }

    }*/




template <class uword>
void HybridBitmap<uword>::xorNot(const HybridBitmap &a, HybridBitmap &container) const{

    //double expDens = (this.setbits+a.setbits)/(double)(this.sizeinbits)-(this.setbits*a.setbits)/(double)(this.sizeinbits*a.sizeinbits);
    //double expDens = this.setbits/(double)this.sizeinbits*(a.sizeinbits-a.setbits)/(double)a.sizeinbits+a.setbits/(double)a.sizeinbits*(this.sizeinbits-this.setbits)/(double)this.sizeinbits;
    container.density=density*(a.density)+(1-a.density)*(1-density);
    //    container.sizeinbits=this.sizeinbits;
    if (verbatim && a.verbatim) {
        xorNotVerbatim(a, container);
    }else if(verbatim || a.verbatim){
        xorNotHybrid(a, container);
    }else{
        //container.reserve(.actualsizeinwords + a.actualsizeinwords);
        logicalxornot(a, container);
    }
//		container.age = Math.max(this.age, a.age)+1;
//		if(container.age>20){
//			container.density=container.cardinality();
//		container.age=0;
//		}

}
template <class uword>
void HybridBitmap<uword>::orAndNotV(const HybridBitmap &a, const HybridBitmap &b, HybridBitmap &container) const{
    container.reset();
    container.buffer.reserve(bufferSize());
    container.verbatim = true;
    double multDens = a.density*(1-b.density);
    container.density = density+multDens - (density*multDens);
    for(int i=0; i<buffer.size();i++){
        container.buffer.push_back(buffer[i]|(a.buffer[i]&(~b.buffer[i])));
    }
    // container.actualsizeinwords=this.actualsizeinwords;
    container.sizeinbits = sizeinbits;

}



template <class uword>
void HybridBitmap<uword>::orAndV(const HybridBitmap &a, const HybridBitmap &b, HybridBitmap &container) const{
    container.reset();
    container.buffer.reserve(bufferSize());
    container.verbatim = true;
    double andDens = a.density*b.density;
    container.density = density+andDens-(density*andDens);
    for(int i=0; i<buffer.size();i++){
        container.buffer.push_back( buffer[i]|(a.buffer[i]&b.buffer[i]));
    }
    //   container.actualsizeinwords=this.actualsizeinwords;
    container.sizeinbits = sizeinbits;

}


template <class uword>
void HybridBitmap<uword>::andVerbatim(const HybridBitmap &a, const HybridBitmap &b, HybridBitmap &container) const{
    //container.buffer.reserve(bufferSize());
    container.density = density*a.density*b.density;
    for(int i=0; i<buffer.size();i++){
        container.buffer.push_back(buffer[i]& a.buffer[i]&b.buffer[i]);
    }
    //container.actualsizeinwords=actualsizeinwords;
    container.sizeinbits =  sizeinbits;

}


template <class uword>
void HybridBitmap<uword>::andVerbatim(const HybridBitmap &a, HybridBitmap &container) const{
    container.reset();
    container.density = density*a.density;
    container.verbatim = true;
    container.buffer.reserve(bufferSize()
                             );
    for(int i=0; i<buffer.size();i++){
        container.buffer.push_back(buffer[i] & a.buffer[i]);
    }
    //container.actualsizeinwords=actualsizeinwords;
    container.sizeinbits =  sizeinbits;
    
}


template <class uword>
void HybridBitmap<uword>::orVerbatim(const HybridBitmap &a, HybridBitmap &container) const{
    container.reset();
    container.verbatim = true;
    container.density= (density+a.density)-(density*a.density);
    container.buffer.reserve(bufferSize());
    for (int i = 0; i < buffer.size(); i++) {
        container.buffer.push_back(buffer[i] | a.buffer[i]);
    }

    //container.sizeinbits = this.actualsizeinwords <<6;
    container.sizeinbits = std::max(sizeinbits, a.sizeinbits);

}

template <class uword>
void HybridBitmap<uword>::orHybridCompress(const HybridBitmap &a, HybridBitmap &container) const {
    int j = 0;
    int i = 0;

    if (verbatim) { // this is verbatim
        container.buffer.reserve(bufferSize());
      //  a.rlw = new RunningLengthWord(a.buffer);
        ConstRunningLengthWord<uword> rlwa(a.buffer[0]);
        size_t lastrlwa = 0;
        while (i <buffer.size()) {
            if (rlwa.getRunningBit()) { // fill of ones
                container.addStreamOfEmptyWords(true, rlwa.getRunningLength());
                i += rlwa.getRunningLength();

            } else {
                for (j = 0; j < rlwa.getRunningLength(); j++) {
                    container.add(buffer[i]);
                    i++;
                }
            }

            for (j = 0; j < rlwa.getNumberOfLiteralWords(); j++) {
                container.add(buffer[i] | a.buffer[lastrlwa + j + 1]);
                i++;
            }
            lastrlwa += rlwa.getNumberOfLiteralWords() + 1;
            if(lastrlwa >= a.bufferSize()){
                break;
            }
            rlwa = (a.buffer[lastrlwa]);
        }
    }
    else { // a is verbatim
        container.buffer.reserve(a.bufferSize());
        ConstRunningLengthWord<uword> rlw(buffer[0]);
        size_t lastrlw = 0;
        while (i < a.bufferSize()) {
            if (rlw.getRunningBit()) { // fill of ones
                container.addStreamOfEmptyWords(true, rlw.getRunningLength());
                i += rlw.getRunningLength();

            } else {
                for (j = 0; j < rlw.getRunningLength(); j++) {
                    container.add(a.buffer[i]);
                    i++;
                }
            }

            for (j = 0; j < rlw.getNumberOfLiteralWords(); j++) {
                container.add(a.buffer[i] | buffer[lastrlw + j + 1]);
                i++;
            }
            lastrlw += rlw.getNumberOfLiteralWords() + 1;
            if(lastrlw >= bufferSize()){
                break;
            }
            rlw = (a.buffer[lastrlw]);
        }
    }
    container.sizeinbits = std::max(sizeinbits, a.sizeinbits);
     

}

template <class uword>
void HybridBitmap<uword>::andHybridCompress(const HybridBitmap &a, HybridBitmap &container) const {
    int j = 0;
    int i=0;
    container.density=density*a.density;
    
    if (verbatim) { // this is verbatim
        container.buffer.reserve(a.bufferSize());
        ConstRunningLengthWord<uword> rlwa(a.buffer[0]);
        size_t lastrlwa = 0;
        //BufferedRunningLengthWord<uword> &rlwa = new BufferedRunningLengthWord<uword>(a.buffer, 0);
        while (i < buffer.size()) {
            if (rlwa.getRunningBit()) { // fill of ones
                for (j = 0; j < rlwa.getRunningLength(); j++) {

                    container.add(buffer[i]);
                    //container.setbits+=Long.bitCount(newdata);
                    i++;
                }
            } else {
                container.addStreamOfEmptyWords(false, rlwa.getRunningLength());
                i+=rlwa.getRunningLength();
            }

            for( j=0; j<rlwa.getNumberOfLiteralWords(); j++){

                container.add(buffer[i] & a.buffer[lastrlwa + j + 1]);
                //container.setbits+=Long.bitCount(newdata);
                i++;
            }
            lastrlwa +=rlwa.getNumberOfLiteralWords()+1;
            if(lastrlwa >= a.bufferSize()){
                break;
            }
            rlwa = (a.buffer[lastrlwa]);
            //rlwa.position += rlwa.getNumberOfLiteralWords() + 1;

        }
    } else { // a is verbatim
        ConstRunningLengthWord<uword> rlw(buffer[0]);
        size_t lastrlw = 0;
        container.buffer.reserve(bufferSize());
        //rlw = new RunningLengthWord(this.buffer, 0);

        while (i < a.buffer.size()) {
            if (rlw.getRunningBit()) { // fill of ones
                for (j = 0; j < rlw.getRunningLength(); j++) {
                    container.add(a.buffer[i]);
                    //container.setbits+=Long.bitCount(newdata);
                    i++;
                }
            } else {
                container.addStreamOfEmptyWords(false, rlw.getRunningLength());
                i+=rlw.getRunningLength();
            }

            for( j=0; j<rlw.getNumberOfLiteralWords(); j++){

                container.add(a.buffer[i] & buffer[lastrlw + j + 1]);
                //container.setbits+=Long.bitCount(newdata);
                i++;
            }
            lastrlw +=rlw.getNumberOfLiteralWords()+1;
            if(lastrlw >= bufferSize()){
                break;
            }
            rlw = (buffer[lastrlw]);
            //rlw.position += rlw.getNumberOfLiteralWords() + 1;

        }
    }

}
template <class uword>
void HybridBitmap<uword>::xorVerbatim(const HybridBitmap &a, HybridBitmap &container)const{
  //  container.reserve(this.actualsizeinwords);
    double density = container.density;
    container.reset();
    container.buffer.reserve(bufferSize());
    container.verbatim = true;
    container.density = density;

    for (size_t i = 0; i < buffer.size(); i++) {
        container.buffer.push_back( buffer[i] ^ a.buffer[i] );
    }

   // container.actualsizeinwords = this->actualsizeinwords;
    //container.sizeinbits = this.actualsizeinwords <<6;
    container.sizeinbits = std::max(sizeinbits, a.sizeinbits);
}


template <class uword>
void HybridBitmap<uword>::inPlaceXorVerbatim(const HybridBitmap &a){

    for (int i = 0; i < buffer.size(); i++) {
        buffer[i]=buffer[i]^a.buffer[i];
    }

}

template <class uword>
void HybridBitmap<uword>::andNotHybridCompress(const HybridBitmap &a, HybridBitmap &container)const {
    int j = 0;
    int i=0;
    container.verbatim=false;
    

    if (verbatim) { // this is verbatim
        //container.reserve(a.actualsizeinwords);
        container.buffer.reserve(a.bufferSize());
        ConstRunningLengthWord<uword> rlwa(a.buffer[0]);
        size_t lastrlwa = 0;
       //a.rlw = new BufferedRunningLengthWord<uword>(a.buffer, 0);

        while (i < buffer.size()) {
            if (rlwa.getRunningBit()) { // fill of ones
                
                //container.addStreamOfEmptyWords(false, a.rlw.getRunningLength());
                for (j = 0; j < rlwa.getRunningLength(); j++) {
                    container.add(0);
                    i++;
                }

            } else {
                for (j = 0; j < rlwa.getRunningLength(); j++) {
                    container.add(buffer[i]);
                    i++;
                }

            }
            for( j=0; j<rlwa.getNumberOfLiteralWords(); j++){
                container.add(buffer[i] & (~a.buffer[lastrlwa + j + 1]));
                i++;
            }
            lastrlwa +=rlwa.getNumberOfLiteralWords()+1;
            if(lastrlwa >= a.bufferSize()){
                break;
            }
            rlwa = (a.buffer[lastrlwa]);
           // rlwa.position += rlwa.getNumberOfLiteralWords() + 1;
        }
    } else { // a is verbatim
        container.buffer.reserve(bufferSize());
        ConstRunningLengthWord<uword> rlw(buffer[0]);
        size_t lastrlw = 0;
        while (i < a.bufferSize()) {
            if (rlw.getRunningBit()) { // fill of ones
                for (j = 0; j < rlw.getRunningLength(); j++) {
                    container.add(~a.buffer[i]);
                    i++;
                }
            } else {
                container.addStreamOfEmptyWords(false, rlw.getRunningLength());
                i+=rlw.getRunningLength();
            }
            for( j=0; j<rlw.getNumberOfLiteralWords(); j++){
                container.add((buffer[lastrlw + j + 1])& (~a.buffer[i]));
                i++;
            }
            lastrlw +=rlw.getNumberOfLiteralWords()+1;
            if(lastrlw >= bufferSize()){
                break;
            }
            rlw = (buffer[lastrlw]);
            //rlw.position += rlw.getNumberOfLiteralWords() + 1;
        }
    }
    container.sizeinbits = std::max(sizeinbits, a.sizeinbits);
}




// ---------------------------------------------------------------------------------------------------------------------------------------






template <class uword>
void HybridBitmap<uword>::logicalAndDecompress(const HybridBitmap &a, HybridBitmap &container)const {
   
    container.buffer.reserve(((sizeinbits/sizeof(uword)*8)+1));
    container.verbatim=true;
//    if (RESERVEMEMORY)
//        container.buffer.reserve(buffer.size() > a.buffer.size() ? buffer.size() : a.buffer.size());
    HybridBitmapRawIterator<uword> i = a.raw_iterator();
    HybridBitmapRawIterator<uword> j = raw_iterator();
    uword actualsizeinwords = 0;
    
    if (!(i.hasNext() and j.hasNext())) { // hopefully this never happens...
        container.setSizeInBits(sizeInBits());
        return;
    }
    // at this point, this should be safe:
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    BufferedRunningLengthWord<uword> &rlwj = j.next();
    
    while ((rlwi.size() > 0) && (rlwj.size() > 0)) {
        while ((rlwi.getRunningLength() > 0) || (rlwj.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwj.getRunningLength();
            BufferedRunningLengthWord<uword> &prey(i_is_prey ? rlwi : rlwj);
            BufferedRunningLengthWord<uword> &predator(i_is_prey ? rlwj : rlwi);
            if (!predator.getRunningBit()) {
                std::fill(container.buffer.begin()+actualsizeinwords,container.buffer.begin()+actualsizeinwords+predator.getRunningLength(), ~0L);
                actualsizeinwords+=predator.getRunningLength();
                prey.discardFirstWordsWithReload(predator.getRunningLength());
            } else {
                const size_t index = prey.discharge(container, predator.getRunningLength());
                //container.fastaddStreamOfEmptyWords(false, predator.getRunningLength() -index);
                std::fill(container.buffer.begin()+actualsizeinwords,
                          container.buffer.begin()+actualsizeinwords+predator.getRunningLength()-index, 0);
                actualsizeinwords+=predator.getRunningLength()-index;
            }
             predator.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwj.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                container.addWord(rlwi.getLiteralWordAt(k) & rlwj.getLiteralWordAt(k));
            }
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwj.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    container.setSizeInBits(sizeInBits());
    container.setSizeInBits(sizeInBits() > a.sizeInBits() ? sizeInBits() : a.sizeInBits());
}

template <class uword>
void HybridBitmap<uword>::andHybrid(const HybridBitmap &a, HybridBitmap &container)const {
    container.verbatim=true;
    int j = 0;
    int i=0;
    int runLength=0;

    if (verbatim) { // this is verbatim
        container.buffer.reserve(bufferSize());
        ConstRunningLengthWord<uword> rlwa(a.buffer[0]);
        size_t lastrlwa = 0;
        while (i < bufferSize()) {
            runLength=(int) rlwa.getRunningLength();
            if (rlwa.getRunningBit()) { // fill of ones
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back(buffer[i]);
                    i++;
                }
            } else {
                //container.addStreamOfEmptyWords(false, a.rlw.getRunningLength());
                //i,i+runLength
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back(0);
                }
               // std::fill(container.buffer.begin(),container.buffer.end(),0);
                i+=runLength;
            }
            for(j=0; j<rlwa.getNumberOfLiteralWords(); j++){
                container.buffer.push_back(buffer[i] & a.buffer[lastrlwa+j+1]);
                i++;
            }
            lastrlwa += rlwa.getNumberOfLiteralWords() + 1;
            if(lastrlwa >= a.bufferSize()){
                break;
            }
            rlwa = (a.buffer[lastrlwa]);
            
        }
    } else { // a is verbatim
        container.buffer.reserve(a.bufferSize());
        ConstRunningLengthWord<uword> rlw(buffer[0]);
        size_t lastrlw = 0;
        while (i < a.bufferSize()) {
            runLength=(int) rlw.getRunningLength();
            if (rlw.getRunningBit()) { // fill of ones
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back(a.buffer[i]);
                    i++;
                }
            } else {
                //container.addStreamOfEmptyWords(false, a.rlw.getRunningLength());
                //i,i+runLength
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back(0);
                    i++;
                }
               // std::fill(container.buffer.begin(),container.buffer.end(),0);
                i+=runLength;
            }
            for( j=0; j<rlw.getNumberOfLiteralWords(); j++){
                container.buffer.push_back(a.buffer[i] & buffer[lastrlw+j+1]);
                i++;
            }
            lastrlw +=rlw.getNumberOfLiteralWords()+1;
            if(lastrlw >= bufferSize()){
                break;
            }
            rlw = (buffer[lastrlw]);
        }
    }
    //container.actualsizeinwords=i;
    //container.sizeinbits=container.actualsizeinwords<<6;
    container.sizeinbits = std::max(sizeinbits, a.sizeinbits);
}

template <class uword>
void HybridBitmap<uword>::andNotVerbatim(const HybridBitmap &a, HybridBitmap &container)const{
    container.buffer.reserve(bufferSize());
    container.verbatim = true;
    container.density=density*(1-a.density);
    for (int i = 0; i < bufferSize(); i++) {
        container.buffer.push_back(buffer[i] & (~a.buffer[i]));
    }
    //container.actualsizeinwords = this.actualsizeinwords;
    //container.sizeinbits = this.actualsizeinwords <<6;
    container.sizeinbits = std::max(sizeinbits, a.sizeinbits);
}




template <class uword>
void HybridBitmap<uword>::andNotHybrid(const HybridBitmap &a, HybridBitmap &container)const {
    container.verbatim = true;
    int j = 0;
    int i = 0;
    int runLength = 0;
    if (verbatim) { // this is verbatim
        container.buffer.reserve(bufferSize());
        ConstRunningLengthWord<uword> rlwa(a.buffer[0]);
        size_t lastrlwa = 0;
        while (i < bufferSize()) {
            runLength=(int) rlwa.getRunningLength();
            if (rlwa.getRunningBit()) { // fill of ones
                //std::fill(container.buffer.begin()+i,container.buffer.begin()+i+runLength,0);
                for (int i = 0; i < bufferSize(); i++) {
                    container.buffer.push_back(0);
                    i++;
                }
                //Arrays.fill(container.buffer, i, i+runLength, 0);
                
            } else {
                //container.addStreamOfEmptyWords(false, a.rlw.getRunningLength());
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back(buffer[i]);
                   // container.buffer[i]=buffer[i];
                    i++;
                }
            }
            
            for( j=0; j<rlwa.getNumberOfLiteralWords(); j++){
                container.buffer.push_back(buffer[i] & (~a.buffer[lastrlwa+j+1]));
                //container.buffer[i]=buffer[i]& (~a.buffer[lastrlwa+j+1]);
                i++;
            }
            //rlwa.position += rlwa.getNumberOfLiteralWords() + 1;
            lastrlwa += rlwa.getNumberOfLiteralWords() + 1;
            if(lastrlwa >= a.bufferSize()){
                break;
            }
            rlwa = (a.buffer[lastrlwa]);
            
        }
    } else { // a is verbatim
        container.buffer.reserve(a.bufferSize());
        ConstRunningLengthWord<uword> rlw(buffer[0]);
        size_t lastrlw = 0;
        while (i < a.bufferSize()) {
            runLength=(int) rlw.getRunningLength();
            if (rlw.getRunningBit()) { // fill of ones
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back(~a.buffer[i]);
                    //container.buffer[i]=(~a.buffer[i]);
                    i++;
                }
            } else {
                //container.addStreamOfEmptyWords(false, a.rlw.getRunningLength());
                for (int i = 0; i < a.bufferSize(); i++) {
                    container.buffer.push_back(0);
                    i++;
                }
                //std::fill(container.buffer.begin()+i,container.buffer.begin()+i+runLength,0);
                //Arrays.fill(container.buffer, i, i+runLength, 0);
                //i+=runLength;
            }
            
            for( j=0; j<rlw.getNumberOfLiteralWords(); j++){
                container.buffer.push_back(buffer[lastrlw+j+1]& (~a.buffer[i]));
                //container.buffer[i]=buffer[lastrlw+j+1]& (~a.buffer[i]);
                i++;
            }
            //rlw.position += rlw.getNumberOfLiteralWords() + 1;
            lastrlw += rlw.getNumberOfLiteralWords() + 1;
            if(lastrlw >= bufferSize()){
                break;
            }
            rlw = (buffer[lastrlw]);
            
        }
    }
    
    //container.actualsizeinwords=i;
    //container.sizeinbits=container.actualsizeinwords<<6;
    container.sizeinbits = std::max(sizeinbits, a.sizeinbits);
    
}
//    if (verbatim) { // this is verbatim
//        container.buffer.reserve(bufferSize());
//
//        while (i < bufferSize()) {
//            runLength = (int) rlwa.getRunningLength();
//            if (rlwa.getRunningBit()) { // fill of ones
//                //i + runLength
//                for(j=0;j<runLength;j++){
//                    container.buffer[i]=~0L;
//                    i++;
//                }
//            i += runLength;
//            }
//        }
//    }
//

template <class uword>
void HybridBitmap<uword>::orHybrid(const HybridBitmap &a, HybridBitmap &container)const {
    container.reset();
    container.density= (density+a.density)-(density*a.density);
    container.verbatim=true;
    int j = 0; //used as counter in various places
    int i=0; //position of the container buffer
    int runLength=0;

    if (verbatim) { // this is verbatim
        container.buffer.reserve(bufferSize());
        ConstRunningLengthWord<uword>  rlwa(a.buffer[0]);
        size_t lastrlwa = 0;
        
        while (i < bufferSize()) {
            runLength = rlwa.getRunningLength();
            if (rlwa.getRunningBit()) { // fill of ones
                for(j=0;j<runLength;j++){
                    container.buffer.push_back(~0L);
                    i++;
                }
            } else {
                //container.addStreamOfEmptyWords(false, a.rlw.getRunningLength());
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back(buffer[i]);
                    i++;
                }
            }
            for( j=0; j<rlwa.getNumberOfLiteralWords(); j++){
                container.buffer.push_back(buffer[i]| a.buffer[lastrlwa+j+1]);
                i++;
            }
            lastrlwa += rlwa.getNumberOfLiteralWords()+1;
            if(lastrlwa >= a.bufferSize()){
                break;
            }
            rlwa = (a.buffer[lastrlwa]);
        }
    } else { // a is verbatim
        container.buffer.reserve(a.bufferSize());
        ConstRunningLengthWord<uword>  rlw(buffer[0]);
         size_t lastrlw = 0;
        
        while (i < a.bufferSize()) {
            runLength=rlw.getRunningLength();
            if (rlw.getRunningBit()) { // fill of ones
					for (j = 0; j < runLength; j++) {
						container.buffer.push_back(buffer[i]);
						i++;
					}
            } else {
                //container.addStreamOfEmptyWords(false, a.rlw.getRunningLength());
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back(a.buffer[i]);
                    i++;
                }
            }

            for( j=0; j<rlw.getNumberOfLiteralWords(); j++){
                container.buffer.push_back(a.buffer[i] | buffer[lastrlw+j+1]);
                i++;
            }
            lastrlw +=rlw.getNumberOfLiteralWords()+1;
            if(lastrlw >= bufferSize()){
                break;
            }
            rlw = (buffer[lastrlw]);
            //this.rlw.position += this.rlw.getNumberOfLiteralWords() + 1;
        }
    }
    //container.actualsizeinwords=i;
    //container.sizeinbits=container.actualsizeinwords<<6;
    container.sizeinbits = std::max(sizeinbits, a.sizeinbits);

}

template <class uword>
void HybridBitmap<uword>::xorVerbatimCompress(const HybridBitmap &a, HybridBitmap &container)const {
    container.buffer.reserve(bufferSize());
    container.verbatim = false;
    for (int i = 0; i < bufferSize(); i++) {
        container.add(buffer[i] ^ a.buffer[i]);
    }
}


template <class uword>
void HybridBitmap<uword>::xorNotVerbatim(const HybridBitmap &a, HybridBitmap &container)const {
    container.reset();
    container.buffer.reserve(bufferSize());
    container.density=density*(a.density)+(1-a.density)*(1-density);
    container.verbatim = true;
    
    for (int i = 0; i < bufferSize(); i++) {
        container.buffer.push_back(buffer[i] ^ (~a.buffer[i]));
    }
    
    //container.buffer.rse = bufferSize();
    //container.sizeinbits = this.actualsizeinwords <<6;
    container.sizeinbits = std::max(sizeinbits, a.sizeinbits);
}


template <class uword>
void HybridBitmap<uword>::xorHybridCompress(const HybridBitmap &a, HybridBitmap &container)const {
    int j = 0;
    int i = 0;
    
    if (verbatim) { // this is verbatim
        container.buffer.reserve(bufferSize());
        ConstRunningLengthWord<uword> rlwa(a.buffer[0]);
        size_t lastrlwa = 0;
        //a.rlw = new RunningLengthWord(a.buffer, 0);
        
        while (i < bufferSize()) {
            if (rlwa.getRunningBit()) { // fill of ones
                for (j = 0; j < rlwa.getRunningLength(); j++) {
                    container.add(~buffer[i]);
                    i++;
                }
            } else {
                
                for (j = 0; j < rlwa.getRunningLength(); j++) {
                    container.add(buffer[i]);
                    i++;
                }
            }
            
            for (j = 0; j < rlwa.getNumberOfLiteralWords(); j++) {
                container.add(buffer[i] ^ a.buffer[lastrlwa + j + 1]);
                i++;
            }
            lastrlwa +=rlwa.getNumberOfLiteralWords()+1;
            if(lastrlwa >= a.bufferSize()){
                break;
            }
            rlwa = (a.buffer[lastrlwa]);
           // a.rlw.position += a.rlw.getNumberOfLiteralWords() + 1;
            
        }
    } else { // a is verbatim
        container.buffer.reserve(a.bufferSize());
        ConstRunningLengthWord<uword> rlw(a.buffer[0]);
        size_t lastrlw = 0;
        while (i < a.bufferSize()) {
            if (rlw.getRunningBit()) { // fill of ones
                for (j = 0; j < rlw.getRunningLength(); j++) {
                    container.add(~a.buffer[i]);
                    i++;
                }
            } else {
                for (j = 0; j < rlw.getRunningLength(); j++) {
                    container.add(a.buffer[i]);
                    i++;
                }
            }
            
            for (j = 0; j < rlw.getNumberOfLiteralWords(); j++) {
                container.add(a.buffer[i] ^ buffer[lastrlw + j + 1]);
                i++;
            }
            
            lastrlw += rlw.getNumberOfLiteralWords()+1;
            if(lastrlw >= bufferSize()){
                break;
            }
            rlw = (buffer[lastrlw]);
            //rlw.position += rlw.getNumberOfLiteralWords() + 1;
        }
    }
    container.sizeinbits = std::max(sizeinbits, a.sizeinbits);
}


template <class uword>
void HybridBitmap<uword>::xorHybrid(const HybridBitmap &a, HybridBitmap &container)const {
    container.reset();
    container.verbatim=true;
    container.density=density*(1-a.density)+a.density*(1-density);
    int j = 0;
    int i=0;
    int runLength=0;
    
    
    
    if (verbatim) { // this is verbatim
        container.buffer.reserve(bufferSize());
        ConstRunningLengthWord<uword> rlwa(a.buffer[0]);
        size_t lastrlwa = 0;
        //a.rlw = new RunningLengthWord(a.buffer, 0);
        
        while (i < bufferSize()) {
            runLength=(int) rlwa.getRunningLength();
            if (rlwa.getRunningBit()) { // fill of ones
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back(~(buffer[i]));
                    i++;
                }
            } else {
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back((buffer[i]));
                    i++;
                }
            }
            
            for( j=0; j<rlwa.getNumberOfLiteralWords(); j++){
                container.buffer.push_back(buffer[i]^(a.buffer[lastrlwa+j+1]));
                i++;
            }
            
            lastrlwa +=rlwa.getNumberOfLiteralWords()+1;
            if(lastrlwa >= a.bufferSize()){
                break;
            }
            rlwa = (a.buffer[lastrlwa]);
           // a.rlw.position += a.rlw.getNumberOfLiteralWords() + 1;
            
            
        }
    } else { // a is verbatim
        container.buffer.reserve(a.bufferSize());
        ConstRunningLengthWord<uword> rlw(buffer[0]);
        size_t lastrlw = 0;
        
        while (i < a.bufferSize()) {
            runLength=(int) rlw.getRunningLength();
            if (rlw.getRunningBit()) { // fill of ones
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back(~(a.buffer[i]));
                    i++;
                }
            } else {
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back(a.buffer[i]);
                    i++;
                }
            }
            
            for( j=0; j<rlw.getNumberOfLiteralWords(); j++){
                container.buffer.push_back(a.buffer[i]^(buffer[lastrlw+j+1]));
                i++;
            }
            lastrlw +=rlw.getNumberOfLiteralWords()+1;
            if(lastrlw >= bufferSize()){
                break;
            }
            rlw = (buffer[lastrlw]);
        }
    }
   // container.actualsizeinwords=i;
    //container.sizeinbits=container.actualsizeinwords<<6;
    container.sizeinbits = std::max(sizeinbits, a.sizeinbits);
}


template <class uword>
void HybridBitmap<uword>::inPlaceXorHybrid(const HybridBitmap &a) {

    int j = 0;
    int i=0;
    int runLength=0;

    if (verbatim) { // this is verbatim
        ConstRunningLengthWord<uword> rlwa(a.buffer[0]);
        size_t lastrlwa = 0;
        //a.rlw = new RunningLengthWord(a.buffer, 0);

        while (i < bufferSize()) {
            runLength=(int) rlwa.getRunningLength();
            if (rlwa.getRunningBit()) { // fill of ones
                for (j = 0; j < runLength; j++) {
                    //container.buffer.push_back(~(buffer[i]));
                    buffer[i]=~buffer[i];
                    i++;
                }
            } else {
                    i=i+runLength;
            }
            for( j=0; j<rlwa.getNumberOfLiteralWords(); j++){
                buffer[i]=buffer[i]^(a.buffer[lastrlwa+j+1]);
                i++;
            }
            lastrlwa +=rlwa.getNumberOfLiteralWords()+1;
            if(lastrlwa >= a.bufferSize()){
                break;
            }
            rlwa = (a.buffer[lastrlwa]);
        }
    } else { // a is verbatim
        ConstRunningLengthWord<uword> rlw(buffer[0]);
        size_t lastrlw = 0;

        while (i < a.bufferSize()) {
            runLength=(int) rlw.getRunningLength();
            if (rlw.getRunningBit()) { // fill of ones
                for (j = 0; j < runLength; j++) {
                    buffer[i]=~(a.buffer[i]);
                    i++;
                }
            } else {
                for (j = 0; j < runLength; j++) {
                    buffer[i]=(a.buffer[i]);
                    i++;
                }
            }

            for( j=0; j<rlw.getNumberOfLiteralWords(); j++){
                buffer[i]=(a.buffer[i]^(buffer[lastrlw+j+1]));
                i++;
            }
            lastrlw +=rlw.getNumberOfLiteralWords()+1;
            if(lastrlw >= bufferSize()){
                break;
            }
            rlw = (buffer[lastrlw]);
        }
    }
    // container.actualsizeinwords=i;
    //container.sizeinbits=container.actualsizeinwords<<6;
    //container.sizeinbits = std::max(sizeinbits, a.sizeinbits);
}


template <class uword>
void HybridBitmap<uword>::xorNotHybrid(const HybridBitmap &a, HybridBitmap &container)const{
    container.reset();
    container.verbatim=true;
    container.density=density*(a.density)+(1-a.density)*(1-density);
    int j = 0;
    int i=0;
    int runLength=0;
    if (verbatim) { // this is verbatim
        container.buffer.reserve(bufferSize());
        ConstRunningLengthWord<uword> rlwa(a.buffer[0]);
        size_t lastrlwa = 0;
        while (i < bufferSize()) {
            runLength=(int) rlwa.getRunningLength();
            if (rlwa.getRunningBit()) { // fill of ones
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back((buffer[i]));
                    i++;
                }
            } else {
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back(~(buffer[i]));
                    i++;
                }
            }
            for( j=0; j<rlwa.getNumberOfLiteralWords(); j++){
                container.buffer.push_back(buffer[i]^(~a.buffer[lastrlwa+j+1]));
                i++;
            }
            lastrlwa +=rlwa.getNumberOfLiteralWords()+1;
            if(lastrlwa >= a.bufferSize()){
                break;
            }
            rlwa = (a.buffer[lastrlwa]);
        }
    } else { // a is verbatim
        container.buffer.reserve(a.bufferSize());
        ConstRunningLengthWord<uword> rlw(buffer[0]);
        size_t lastrlw = 0;
        while (i < a.bufferSize()) {
            runLength=(int) rlw.getRunningLength();
            if (rlw.getRunningBit()) { // fill of ones
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back(a.buffer[i]);
                    i++;
                }
            } else {
                for (j = 0; j < runLength; j++) {
                    container.buffer.push_back(~(a.buffer[i]));
                    i++;
                }
            }
            
            for( j=0; j<rlw.getNumberOfLiteralWords(); j++){
                //                    if((this.rlw.position+j+1)>=this.buffer.length)
                //                        System.out.println("ERROR C has broblems. buffer size: "+this.buffer.length);
                //                    if(i>=a.buffer.length)
                //                        System.out.println("ERROR Verbatim has broblems. Num of bits: "+a.sizeinbits);
                container.buffer.push_back((~a.buffer[i])^(buffer[lastrlw+j+1]));
                i++;
            }
            lastrlw +=rlw.getNumberOfLiteralWords()+1;
            if(lastrlw >= bufferSize()){
                break;
            }
            rlw = (buffer[lastrlw]);
            
        }
        
    }
    
    //container.actualsizeinwords=i;
    //container.sizeinbits=container.actualsizeinwords<<6;
    container.sizeinbits = std::max(sizeinbits, a.sizeinbits);
    
}


template <class uword>
    void HybridBitmap<uword>::logicalxorDecompress(const HybridBitmap &a, HybridBitmap &container)const{
    container.reset();
    container.verbatim=true;
        if (RESERVEMEMORY){
            container.buffer.reserve(buffer.size() + a.buffer.size());}
            
    HybridBitmapRawIterator<uword> i = a.raw_iterator();
    HybridBitmapRawIterator<uword> j = raw_iterator();
    uword actualsizeinwords = 0;
    
    if (!(i.hasNext() and j.hasNext())) { // hopefully this never happens...
        container.setSizeInBits(sizeInBits());
        return;
    }
    // at this point, this should be safe:
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    BufferedRunningLengthWord<uword> &rlwj = j.next();
    while ((rlwi.size() > 0) && (rlwj.size() > 0)) {
        while ((rlwi.getRunningLength() > 0) || (rlwj.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwj.getRunningLength();
            BufferedRunningLengthWord<uword> &prey = i_is_prey ? rlwi : rlwj;
            BufferedRunningLengthWord<uword> &predator = i_is_prey ? rlwj : rlwi;
            
            if (predator.getRunningBit() == false) {
                const size_t index = prey.dischargeDecompressed(container, predator.getRunningLength());
                //container.addStreamOfEmptyWords(false, predator.getRunningLength()
                // - index);
                for(int i = 0; i< predator.getRunningLength() - index; i++){
                    container.buffer.push_back(0);
                }
                
            } else {
                 const size_t index = prey.dischargeNegatedDecompressed(container, predator.getRunningLength());
                // container.addStreamOfEmptyWords(true, predator.getRunningLength()
                //  - index);
                for(int i = 0; i< predator.getRunningLength() - index; i++){
                    container.buffer.push_back(~0L);
                }
            }
//            ? prey.dischargeDecompressed(container, predator.getRunningLength())
//            : prey.dischargeNegatedDecompressed(container, predator.getRunningLength());
//            for(int i = 0; i< predator.getRunningLength() - index; i++){
//                container.buffer.push_back(0);
//            }
//            std::fill(container.buffer.begin()+actualsizeinwords, container.buffer.begin()+actualsizeinwords+predator.getRunningLength()-index, 0);
            actualsizeinwords += predator.getRunningLength();
            predator.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(), rlwj.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k){
                container.buffer.push_back((rlwi.getLiteralWordAt(k) ^ rlwj.getLiteralWordAt(k)));
//                container.addWord(rlwi.getLiteralWordAt(k) ^ rlwj.getLiteralWordAt(k));
                actualsizeinwords++;
            }
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwj.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    const bool i_remains = rlwi.size() > 0;
    BufferedRunningLengthWord<uword> &remaining = i_remains ? rlwi : rlwj;
    remaining.discharge(container);
    container.setSizeInBits(sizeInBits() > a.sizeInBits() ? sizeInBits() : a.sizeInBits());
}

template <class uword>
void HybridBitmap<uword>::add(long newdata, const int bitsthatmatter) {
    sizeinbits += bitsthatmatter;
    if (newdata == 0) {
        addEmptyWord(false);
    } else if (newdata == ~0l) {
        addEmptyWord(true);
    } else {
        addLiteralWord(newdata);
    }
}


template <class uword>
void HybridBitmap<uword>::add(const long newdata) {
    add(newdata, wordinbits);
}


template <class uword>
void HybridBitmap<uword>::logicalxornot(const HybridBitmap &a, HybridBitmap &container) const{
    container.reset();
    container.buffer.push_back(0);
    if (RESERVEMEMORY)
        container.buffer.reserve(buffer.size() + a.buffer.size());
    HybridBitmapRawIterator<uword> i = a.raw_iterator();
    HybridBitmapRawIterator<uword> j = raw_iterator();
    if (!(i.hasNext() and j.hasNext())) { // hopefully this never happens...
        container.setSizeInBits(sizeInBits());
        return;
    }
    // at this point, this should be safe:
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    BufferedRunningLengthWord<uword> &rlwj = j.next();
    while ((rlwi.size() > 0) && (rlwj.size() > 0)) {
        while ((rlwi.getRunningLength() > 0) || (rlwj.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwj.getRunningLength();
            BufferedRunningLengthWord<uword> &prey = i_is_prey ? rlwi : rlwj;
            BufferedRunningLengthWord<uword> &predator = i_is_prey ? rlwj : rlwi;
            const size_t index =
            (!predator.getRunningBit())
            ? prey.discharge(container, predator.getRunningLength())
            : prey.dischargeNegated(container, predator.getRunningLength());
            container.fastaddStreamOfEmptyWords(predator.getRunningBit(),
                                                predator.getRunningLength() - index);
            predator.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwj.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k)
                container.addWord(~(rlwi.getLiteralWordAt(k) ^ rlwj.getLiteralWordAt(k)));
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwj.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    const bool i_remains = rlwi.size() > 0;
    BufferedRunningLengthWord<uword> &remaining = i_remains ? rlwi : rlwj;
    remaining.discharge(container);
    container.setSizeInBits(sizeInBits() > a.sizeInBits() ? sizeInBits() : a.sizeInBits());
}


//template <class uword>
//int HybridBitmap<uword>::xorCardinality(final HybridBitmap a) {
//    BitCounter counter = new BitCounter();
//    logicalxor(a, counter);
//    return counter.getCount();
//}

#endif
