//
//  BsiAttribute.hpp
//

#ifndef BsiAttribute_cpp
#define BsiAttribute_cpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "hybridbitmap.cpp"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <bitset>

//for pytorch
#include <torch/extension.h>


template <class uword = uint64_t> class BsiUnsigned;
template <class uword = uint64_t> class BsiSigned;
//template <class uword = uint32_t>
template <class uword = uint64_t> class BsiAttribute{
public:
    uint8_t size; //holds number of slices
    int offset =0;
    int decimals = 0;
    int bits = 8*sizeof(uword);
    std::vector<HybridBitmap<uword> > bsi ;
    HybridBitmap<uword> existenceBitmap;
    long rows;  // number of elements
    long index; // if split horizontally
    bool is_signed; // sign flag
    bool firstSlice; //contains first slice
    bool lastSlice; //contains last slice
    HybridBitmap<uword> sign; // sign bitslice
    bool twosComplement;
    
    /*Declare functions
     */
    bool isLastSlice() const;
    void setLastSliceFlag(bool flag);
    
    bool isFirstSlice() const;
    void setFirstSliceFlag(bool flag);
    bool isSigned()const;
    void addSlice(const HybridBitmap<uword> &slice);
    void setNumberOfSlices(int s);
    int getNumberOfSlices() const;
    HybridBitmap<uword> getSlice(int i) const;
    int getOffset() const;
    void setOffset(int offset);
  
    
    long getNumberOfRows() const;   // return size of vector of original numbers
    void setNumberOfRows(long rows);
    long getPartitionID() const;
    void setPartitionID(long index);
    HybridBitmap<uword> getExistenceBitmap();
    void setExistenceBitmap(const HybridBitmap<uword> &exBitmap);
    void setTwosFlag(bool flag);
    
    virtual HybridBitmap<uword> topKMax(int k)=0;
    virtual HybridBitmap<uword> topKMin(int k)=0;
    virtual BsiAttribute* SUM(BsiAttribute* a)const=0;
    virtual BsiAttribute* SUM(long a)const=0;
    virtual BsiAttribute* convertToTwos(int bits)=0;
    virtual BsiUnsigned<uword>* abs()=0;
    virtual BsiUnsigned<uword>* abs(int resultSlices,const HybridBitmap<uword> &EB)=0;
    virtual BsiUnsigned<uword>* absScale(double range)=0;
    virtual long getValue(int pos)=0;   
    virtual HybridBitmap<uword> rangeBetween(long lowerBound, long upperBound)=0;
    virtual BsiAttribute<uword>* multiplyByConstant(int number)const=0;
    virtual BsiAttribute<uword>* multiplication(BsiAttribute<uword> *a)const=0;
    virtual BsiAttribute<uword>* multiplication_array(BsiAttribute<uword> *a)const=0;
    virtual BsiAttribute<uword>* multiplyBSI(BsiAttribute<uword> *a) const=0;
    virtual void multiplicationInPlace(BsiAttribute<uword> *a)=0;
    virtual BsiAttribute<uword>* negate()=0;
    virtual long sumOfBsi()const=0;
    virtual bool append(long value)=0;
    
    BsiAttribute* buildQueryAttribute(long query, int rows, long partitionID);
    BsiAttribute* buildBsiAttributeFromArray(std::vector<uword> &array, int attRows, double compressThreshold);
    BsiAttribute* buildBsiAttributeFromArray(uword array[], long max, long min, long firstRowID, double compressThreshold);
    BsiAttribute<uword>* buildBsiAttributeFromVector(std::vector<long> nums, double compressThreshold)const;
    BsiAttribute<uword>* buildBsiAttributeFromTensor(torch::Tensor nums_tensor, double compressThreshold) const;
    BsiAttribute<uword>* buildCompressedBsiFromVector(std::vector<long> nums, double compressThreshold) const;
    
    HybridBitmap<uword> maj(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const;
    HybridBitmap<uword> XOR(const HybridBitmap<uword> &a,const HybridBitmap<uword> &b,const HybridBitmap<uword> &c)const;
    HybridBitmap<uword> orAndNot(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const;
    HybridBitmap<uword> orAnd(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const;
    HybridBitmap<uword> And(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const;
   
    BsiAttribute* signMagnToTwos(int bits);
    BsiAttribute* TwosToSignMagnitue();
    
    void signMagnitudeToTwos(int bits);
    void addOneSliceSameOffset(const HybridBitmap<uword> &slice);
    void addOneSliceDiscardCarry(const HybridBitmap<uword> &slice);
    void addOneSliceNoSignExt(const HybridBitmap<uword> &slice);
    void applyExsistenceBitmap(const HybridBitmap<uword> &ex);
    
    virtual ~BsiAttribute();
private:
    std::vector< std::vector< uword > > bringTheBits(const std::vector<long> &array, int slices, int attRows) const;
    std::vector< std::vector< uword > > bringTheBits(const std::vector<uword> &array, int slices, int attRows) const;
    std::vector< std::vector< uword > > bringTheBits(const torch::Tensor &array_tensor, int slices, int attRows) const;
protected:
    int sliceLengthFinder(uword value)const;
     
};


//------------------------------------------------------------------------------------------------------

/*
 * Destructor
 */
template <class uword>
BsiAttribute<uword>::~BsiAttribute(){
    
};


template <class uword>
bool BsiAttribute<uword>::isLastSlice() const{
    return lastSlice;
};

/*
 *
 * @param flag if the attribute contains the most significant slice then set it to true. Otherwise false.
 */
template <class uword>
void BsiAttribute<uword>::setLastSliceFlag(bool flag){
    lastSlice=flag;
};
/*
 *
 * @return if this attribute contains the first slice(least significant). For internal purposes (when splitting into sub attributes)
 */
template <class uword>
bool BsiAttribute<uword>::isFirstSlice()const{
    return firstSlice;
};
/*
 *
 * @param flag if the attribute contains the least significant slice then set it to true. Otherwise false.
 */
template <class uword>
void BsiAttribute<uword>::setFirstSliceFlag(bool flag){
    firstSlice=flag;
};

/*
 * Returns false if contains only positive numbers
 */
template <class uword>
bool BsiAttribute<uword>::isSigned()const{
    return is_signed;
};

template <class uword>
void BsiAttribute<uword>::addSlice( const HybridBitmap<uword> &slice){
    bsi.push_back(slice);
    size++;
};

/*
 * Don't use for already buily bsi
 */

template <class uword>
void BsiAttribute<uword>::setNumberOfSlices(int s){
    size = s;
}

/**
 * Returns the size of the bsi (how many slices are non zeros)
 */
template <class uword>
int BsiAttribute<uword>::getNumberOfSlices()const{
    return bsi.size();
}

/**
 * Returns the slice number i
 */
template <class uword>
HybridBitmap<uword> BsiAttribute<uword>::getSlice(int i) const{
    return bsi[i];
}


/**
 * Returns the offset of the bsi (the first "offset" slices are zero, thus not encoding)
 */
template <class uword>
int BsiAttribute<uword>::getOffset() const{
    return offset;
}


/**
 * Sets the offset of the bsi (the first "offset" slices are zero, thus not encoding)
 */

template <class uword>
void BsiAttribute<uword>::setOffset(int offset){
    BsiAttribute::offset=offset;
}

/**
 * Returns the number of rows for this attribute
 */

template <class uword>
long BsiAttribute<uword>::getNumberOfRows() const{
    return rows;
}


/**
 * Sets the number of rows for this attribute
 */
template <class uword>
void BsiAttribute<uword>::setNumberOfRows(long rows){
    BsiAttribute::rows=rows;
}

/**
 * Returns the index(partition id if horizontally partitioned) for this attribute
 */
template <class uword>
long BsiAttribute<uword>::getPartitionID() const{
    return index;
}

/**
 * Sets the index(partition id if horizontally partitioned) for this attribute
 */
template <class uword>
void BsiAttribute<uword>::setPartitionID(long index){
    BsiAttribute::index=index;
}

/**
 * Returns the Existence bitmap of the bsi attribute
 */
template <class uword>
HybridBitmap<uword> BsiAttribute<uword>::getExistenceBitmap(){
    return existenceBitmap;
}

/**
 * Sets the existence bitmap of the bsi attribute
 */
template <class uword>
void BsiAttribute<uword>::setExistenceBitmap(const HybridBitmap<uword> &exBitmap){
    BsiAttribute::existenceBitmap=exBitmap;
}

/**
 * flag is true when bsi contain data into two's complement form
 */
template <class uword>
void BsiAttribute<uword>::setTwosFlag(bool flag){
    twosComplement=flag;
}

/**
 * builds a BSI attribute with all rows identical given one number (row)
 * @param query
 * @param rows
 * @return the BSI attribute with all rows identical
 */

template <class uword>
BsiAttribute<uword>* BsiAttribute<uword>::buildQueryAttribute(long query, int rows, long partitionID){
    if(query<0){
        uword q = std::abs(query);
        int maxsize = sliceLengthFinder(q);
        BsiAttribute* res = new BsiSigned<uword>(maxsize);
        res->setPartitionID(partitionID);
        for(int i=0; i<=maxsize; i++){
            bool currentBit = (q&(1<<i))!=0;
            HybridBitmap<uword> slice;
            slice.setSizeInBits(rows, currentBit);
            if(currentBit){
                slice.density = 1;
            }
            res->addSlice(slice);
        }
        res->setNumberOfRows(rows);
        res->existenceBitmap.setSizeInBits(rows);
        res->existenceBitmap.density=1;
        res->lastSlice=true;
        res->firstSlice=true;
        res->twosComplement=false;
        res->is_signed = true;
        HybridBitmap<uword> temp_sign(true,rows);
        res->sign = temp_sign.Not();//set the sign bits true
        return res;
    }
    else{
        int maxsize = sliceLengthFinder(query);
        BsiAttribute* res = new BsiUnsigned<uword>(maxsize);
        res->setPartitionID(partitionID);
        for(int i=0; i<=maxsize; i++){
            bool currentBit = (query&(1<<i))!=0;
            HybridBitmap<uword> slice;
            slice.setSizeInBits(rows, currentBit);
            if(currentBit){
                slice.density = 1;
            }
            res->addSlice(slice);
        }
        res->setNumberOfRows(rows);
        res->existenceBitmap.setSizeInBits(rows,true);
        res->existenceBitmap.density=1;
        res->lastSlice=true;
        res->firstSlice=true;
        return res;
    }
};

/*
 *
 * sliceLengthFinder find required slices for storing value
 */
template <class uword>
int BsiAttribute<uword>::sliceLengthFinder(uword value) const{
    int lengthCounter =0;
    for(int i = 0; i < bits; i++)
    {
        //uword ai = (static_cast<uword>(1) << i);
        if( ( value & (static_cast<uword>(1) << i ) ) != 0 ){
            lengthCounter = i+1;
        }
    }
    return lengthCounter;
}

/*
 * Used for converting vector into BSI
 * @param compressThreshold determined wether to compress the bit vetor or not
 */


template <class uword>
BsiAttribute<uword>* BsiAttribute<uword>::buildBsiAttributeFromVector(std::vector<long> nums, double compressThreshold) const{
    uword max = std::numeric_limits<uword>::min();
    
    int attRows = nums.size();
    std::vector<uword> signBits(attRows/(bits)+1);
    std::vector<uword> existBits(attRows/(bits)+1); // keep track for non-zero values
    int countOnes =0;
    int CountZeros = 0;
    //find max, min, and zeros.
    for (int i=0; i<nums.size(); i++){
        int offset = i%(bits);
        if(nums[i] < 0){
            nums[i] = 0 - nums[i];
            signBits[i / (bits)] |= (1L << offset); // seting sign bit
            countOnes++;
        }
        if(nums[i] != 0){
            existBits[i / (bits)] |= (1L << offset); // seting one at position
        }else{
            CountZeros++;
        }
        if(nums[i] > max){
            max = nums[i];
        }
    }
    int slices = sliceLengthFinder(max);
    BsiSigned<uword>* res = new BsiSigned<uword>(slices+1);
    res->sign.reset();
    res->sign.verbatim = true;
    
    for (typename std::vector<uword>::iterator it=signBits.begin(); it != signBits.end(); it++){
        res->sign.addVerbatim(*it,attRows);
    }
    res->sign.setSizeInBits(attRows);
    res->sign.density = countOnes/(double)attRows;
    
    double existBitDensity = (CountZeros/(double)nums.size()); // to decide whether to compress or not
    double existCompressRatio = 1-pow((1-existBitDensity), (2*bits))-pow(existBitDensity, (2*bits));
    if(existCompressRatio >= compressThreshold){
        HybridBitmap<uword> bitmap;
        for(int j=0; j<existBits.size(); j++){
            bitmap.addWord(existBits[j]);
        }
        //bitmap.setSizeInBits(attRows);
        bitmap.density=existBitDensity;
        res->setExistenceBitmap(bitmap);
    }else{
        HybridBitmap<uword> bitmap(true,existBits.size());
        for(int j=0; j<existBits.size(); j++){
            bitmap.buffer[j] = existBits[j];
        }
        //bitmap.setSizeInBits(attRows);
        bitmap.density=existBitDensity;
        res->setExistenceBitmap(bitmap);
    }
    
    std::vector< std::vector< uword > > bitSlices = bringTheBits(nums,slices,attRows);
    
    for(int i=0; i<slices; i++){
        double bitDensity = bitSlices[i][0]/(double)attRows; // the bit density for this slice
        double compressRatio = 1-pow((1-bitDensity), (2*bits))-pow(bitDensity, (2*bits));
        if(compressRatio<compressThreshold){
            //build compressed bitmap
            HybridBitmap<uword> bitmap;
            for(int j=1; j<bitSlices[i].size(); j++){
                bitmap.addWord(bitSlices[i][j]);
            }
            //bitmap.setSizeInBits(attRows);
            bitmap.density=bitDensity;
            res->addSlice(bitmap);

        }else{
            //build verbatim Bitmap
            HybridBitmap<uword> bitmap(true);
            bitmap.reset();
            bitmap.verbatim = true;
            //                std::copy(bitSlices[i].begin(), bitSlices[i].end(), bitmap.buffer.begin());
            for (typename std::vector<uword>::iterator it=bitSlices[i].begin()+1; it != bitSlices[i].end(); it++){
                bitmap.addVerbatim(*it,attRows);
            }
            // bitmap.buffer=Arrays.copyOfRange(bitSlices[i], 1, bitSlices[i].length);
            //bitmap.actualsizeinwords=bitSlices[i].length-1;
            bitmap.setSizeInBits(attRows);
            bitmap.density=bitDensity;
            res->addSlice(bitmap);
            
        }
    }
    res->existenceBitmap.setSizeInBits(attRows,true);
    res->existenceBitmap.density=1;
    res->lastSlice=true;
    res->firstSlice=true;
    res->twosComplement=false;
    res->rows = attRows;
    res->is_signed = true;
    return res;
};

/**
 * Convert torch tensor to bsi attribute
*/

template <class uword>
BsiAttribute<uword>* BsiAttribute<uword>::buildBsiAttributeFromTensor(torch::Tensor nums_tensor, double compressThreshold) const {
    auto nums = nums_tensor.accessor<uword, 1>(); //get accessor

    uword max = std::numeric_limits<uword>::min();
    auto attRows = nums.size(0);

    std::vector<uword> signBits(attRows/(bits)+1);
    std::vector<uword> existBits(attRows/(bits)+1);
    int countOnes =0;
    int CountZeros = 0;

    for(int i=0; i<nums.size(0); i++) {
        int offset = i%(bits);
        if(nums[i] < 0){
            nums[i] = 0 - nums[i];
            signBits[i / (bits)] |= (1L << offset); // seting sign bit
            countOnes++;
        }
        if(nums[i] != 0){
            existBits[i / (bits)] |= (1L << offset); // seting one at position
        }else{
            CountZeros++;
        }
        if(nums[i] > max){
            max = nums[i];
        }
    }

    int slices = sliceLengthFinder(max);
    BsiSigned<uword>* res = new BsiSigned<uword>(slices+1);
    res->sign.reset();
    res->sign.verbatim = true;

    for (typename std::vector<uword>::iterator it=signBits.begin(); it != signBits.end(); it++){
        res->sign.addVerbatim(*it,attRows);
    }
    res->sign.setSizeInBits(attRows);
    res->sign.density = countOnes/(double)attRows;

    double existBitDensity = (CountZeros/(double)nums.size(0)); // to decide whether to compress or not
    double existCompressRatio = 1-pow((1-existBitDensity), (2*bits))-pow(existBitDensity, (2*bits));

    if(existCompressRatio >= compressThreshold){
        HybridBitmap<uword> bitmap;
        for(int j=0; j<existBits.size(); j++){
            bitmap.addWord(existBits[j]);
        }
        //bitmap.setSizeInBits(attRows);
        bitmap.density=existBitDensity;
        res->setExistenceBitmap(bitmap);
    }else{
        HybridBitmap<uword> bitmap(true,existBits.size());
        for(int j=0; j<existBits.size(); j++){
            bitmap.buffer[j] = existBits[j];
        }
        //bitmap.setSizeInBits(attRows);
        bitmap.density=existBitDensity;
        res->setExistenceBitmap(bitmap);
    }

    std::vector< std::vector< uword > > bitSlices = bringTheBits(nums_tensor,slices,attRows);
    
    for(int i=0; i<slices; i++){
        double bitDensity = bitSlices[i][0]/(double)attRows; // the bit density for this slice
        double compressRatio = 1-pow((1-bitDensity), (2*bits))-pow(bitDensity, (2*bits));
        if(compressRatio<compressThreshold){
            //build compressed bitmap
            HybridBitmap<uword> bitmap;
            for(int j=1; j<bitSlices[i].size(); j++){
                bitmap.addWord(bitSlices[i][j]);
            }
            //bitmap.setSizeInBits(attRows);
            bitmap.density=bitDensity;
            res->addSlice(bitmap);

        }else{
            //build verbatim Bitmap
            HybridBitmap<uword> bitmap(true);
            bitmap.reset();
            bitmap.verbatim = true;
            //                std::copy(bitSlices[i].begin(), bitSlices[i].end(), bitmap.buffer.begin());
            for (typename std::vector<uword>::iterator it=bitSlices[i].begin()+1; it != bitSlices[i].end(); it++){
                bitmap.addVerbatim(*it,attRows);
            }
            // bitmap.buffer=Arrays.copyOfRange(bitSlices[i], 1, bitSlices[i].length);
            //bitmap.actualsizeinwords=bitSlices[i].length-1;
            bitmap.setSizeInBits(attRows);
            bitmap.density=bitDensity;
            res->addSlice(bitmap);
            
        }
    }
    res->existenceBitmap.setSizeInBits(attRows,true);
    res->existenceBitmap.density=1;
    res->lastSlice=true;
    res->firstSlice=true;
    res->twosComplement=false;
    res->rows = attRows;
    res->is_signed = true;
    return res;
}


/*
 * private function
 */
template <class uword>
std::vector< std::vector< uword > > BsiAttribute<uword>::bringTheBits(const std::vector<long> &array, int slices, int attRows) const{
    
    int wordsNeeded = ceil( attRows / (double)(bits));
    std::vector< std::vector< uword > > bitmapDataRaw(slices,std::vector<uword>(wordsNeeded +1));
    
    // one for the bit density (the first word in each slice)
    uword thisBin = 0;
    for (int seq = 0; seq < attRows; seq++) {
        int w = (seq / (bits)+1);
                int offset = seq % (bits);
        thisBin = array[seq];
        int slice = 0;
        while (thisBin != 0 && slice<slices) {
            if ((thisBin & 1) == 1) {
                bitmapDataRaw[slice][w] |= (1L << offset); //setting bit
                bitmapDataRaw[slice][0]++; //update bit density
            }
            thisBin >>= 1;
            slice++;
        }
    }
    return bitmapDataRaw;
};

/**
 * tensor version
*/
template <class uword>
std::vector< std::vector< uword > > BsiAttribute<uword>::bringTheBits(const torch::Tensor &array_tensor, int slices, int attRows) const {
    auto array = array_tensor.accessor<uword, 1>();
    
    int wordsNeeded = ceil( attRows / (double)(bits));
    std::vector< std::vector< uword > > bitmapDataRaw(slices,std::vector<uword>(wordsNeeded +1));

    uword thisBin = 0;
    for (int seq = 0; seq < attRows; seq++) {
        int w = (seq / (bits)+1);
                int offset = seq % (bits);
        thisBin = array[seq];
        int slice = 0;
        while (thisBin != 0 && slice<slices) {
            if ((thisBin & 1) == 1) {
                bitmapDataRaw[slice][w] |= (1L << offset); //setting bit
                bitmapDataRaw[slice][0]++; //update bit density
            }
            thisBin >>= 1;
            slice++;
        }
    }
    return bitmapDataRaw;
}

/*
 * maj perform c = a&b | b&c | a&c
 */
template <class uword>
HybridBitmap<uword> BsiAttribute<uword>::maj(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const{
    
    if(a.verbatim && b.verbatim && c.verbatim){
        return a.maj(b, c);
    }else{
        
        return a.logicaland(b).logicalor(b.logicaland(c)).logicalor(a.logicaland(c));
    }
};



template <class uword>
HybridBitmap<uword> BsiAttribute<uword>::XOR(const HybridBitmap<uword> &a,const HybridBitmap<uword> &b,const HybridBitmap<uword> &c)const{
    if(a.verbatim && b.verbatim && c.verbatim){
        return a.Xor(b.Xor(c));
    }else{
        return a.Xor(b).Xor(c);
    }
};

/*
 * perform  a | b & ~c
 */
template <class uword>
HybridBitmap<uword> BsiAttribute<uword>::orAndNot(const HybridBitmap<uword> &a,const HybridBitmap<uword> &b,const HybridBitmap<uword> &c)const{
    if(a.verbatim && b.verbatim && c.verbatim){
        return a.orAndNotV(b,c);
    }else{
        return a.logicalor(b.andNot(c));
    }
};


/*
 * perform  a | b & c
 */
template <class uword>
HybridBitmap<uword> BsiAttribute<uword>::orAnd(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const{
    if(a.verbatim && b.verbatim && c.verbatim){
        return a.orAndV(b,c);
    }else{
        return a.logicalor(b.logicaland(c));
    }
};

template <class uword>
HybridBitmap<uword> BsiAttribute<uword>::And(const HybridBitmap<uword> &a, const HybridBitmap<uword> &b, const HybridBitmap<uword> &c)const{
    if(a.verbatim && b.verbatim && c.verbatim){
        return a.andVerbatim(b,c);
    }else{
        return a.logicaland(b.logicaland(c));
    }
};



/*
 *
 */
template <class uword>
void BsiAttribute<uword>::signMagnitudeToTwos(int bits){
    int i=0;
    for(i=0; i<getNumberOfSlices(); i++){
        bsi[i]=bsi[i].Xor(sign);
    }
    while(i<bits){ // sign extension
        
        addSlice(sign);
        i++;
    }
    if(this->firstSlice){
        addOneSliceSameOffset(sign);
    }
    
    setTwosFlag(true);
};


template <class uword>
BsiAttribute<uword>* BsiAttribute<uword>::signMagnToTwos(int bit_limit){
    BsiAttribute* res = new BsiSigned<uword>();
    res->twosComplement=true;
    int i=0;
    for(i=0; i<getNumberOfSlices(); i++){
        res->bsi[i]=bsi[i].Xor(sign);
    }
    while(i<bit_limit){
        res->addSlice(sign);
        i++;}
    if(firstSlice){
        res->addOneSliceSameOffset(sign);
    }
    return res;
};

template <class uword>
BsiAttribute<uword>* BsiAttribute<uword>::TwosToSignMagnitue(){
    BsiAttribute* res = new BsiSigned<uword>();
    for (int i=0; i<size; i++){
        res->bsi[i]=bsi[i].logicalxor(bsi[size-1]);
    }if(firstSlice){
        res->addOneSliceSameOffset(bsi[size-1]);
    }
    return res;
};


template <class uword>
void BsiAttribute<uword>::addOneSliceSameOffset(const HybridBitmap<uword> &slice){
    HybridBitmap<uword> C,S;
    S=bsi[0].Xor(slice);
    C=bsi[0].And(slice);
    bsi[0]=S;
    int curPos =1;
    while(C.numberOfOnes()>0){
        if(curPos<size){
            S=C.Xor(bsi[curPos]);
            C=C.And(bsi[curPos]);
            bsi[curPos]=S;
            curPos++;
        }else{
            addSlice(C);
            return;
        }
    }
};


template <class uword>
void BsiAttribute<uword>::addOneSliceDiscardCarry(const HybridBitmap<uword> &slice){
    
    HybridBitmap<uword> C,S;
    S=bsi[0].Xor(slice);
    C=bsi[0].And(slice);
    bsi[0]=S;
    int curPos =1;
    while(C.numberOfOnes()>0){
        if(curPos<size){
            S=C.Xor(bsi[curPos]);
            C=C.And(bsi[curPos]);
            bsi[curPos]=S;
            curPos++;
        }
    }
};

template <class uword>
void BsiAttribute<uword>::addOneSliceNoSignExt(const HybridBitmap<uword> &slice){
    
    
    HybridBitmap<uword> C,S;
    S=bsi[0].Xor(slice);
    C=bsi[0].And(slice);
    bsi[0]=S;
    int curPos =1;
    while(C.numberOfOnes()>0){
        if(curPos<size){
            S=C.Xor(bsi[curPos]);
            C=C.And(bsi[curPos]);
            bsi[curPos]=S;
            curPos++;
        }else return;
    }
};

template <class uword>
void BsiAttribute<uword>::applyExsistenceBitmap(const HybridBitmap<uword> &ex){
    existenceBitmap = ex;
    for(int i=0; i< size; i++){
        this->bsi[i] = bsi[i].And(ex);
    }
//    addSlice(ex.Not());
};


/*
 * buildCompressedBsiFromVector is used for making synchronised compressed bsi
 * every bitmap is compressed by same positions
 */


template <class uword>
BsiAttribute<uword>* BsiAttribute<uword>::buildCompressedBsiFromVector(std::vector<long> nums, double compressThreshold) const{
    uword max = std::numeric_limits<uword>::min();
    
    int attRows = nums.size();
    //    int slices = 3*digits + (int)std::log2(digits);
    std::vector<uword> signBits(attRows/(bits)+1);
    std::vector<uword> existBits(attRows/(bits)+1);
    int countOnes =0;
    int CountZeros = 0;
    for (int i=0; i<nums.size(); i++){
        int offset = i%(bits);
        if(nums[i] < 0){
            nums[i] = 0 - nums[i];
            signBits[i / (bits)] |= (1L << offset);
            countOnes++;
        }
        if(nums[i] != 0){
            existBits[i / (bits)] |= (1L << offset);
        }else{
            CountZeros++;
        }
        if(nums[i] > max){
            max = nums[i];
        }
    }
    int slices = sliceLengthFinder(max);
    BsiSigned<uword>* res = new BsiSigned<uword>(slices+1);
    res->sign.reset();
    res->sign.verbatim = true;
    
    for (typename std::vector<uword>::iterator it=signBits.begin(); it != signBits.end(); it++){
        res->sign.addVerbatim(*it,attRows);
    }
    res->sign.setSizeInBits(attRows);
    res->sign.density = countOnes/(double)attRows;
    
    double existBitDensity = 1- (CountZeros/(double)nums.size());
    double existCompressRatio = pow((1-existBitDensity), (2*bits))+pow(existBitDensity, (2*bits));
    if(existCompressRatio >= compressThreshold){
        HybridBitmap<uword> bitmap;
        for(int j=0; j<existBits.size(); j++){
            bitmap.addWord(existBits[j]);
        }
        bitmap.setSizeInBits(nums.size());
        bitmap.density=existBitDensity;
        res->setExistenceBitmap(bitmap);
    }else{
        HybridBitmap<uword> bitmap(true,existBits.size());
        for(int j=0; j<existBits.size(); j++){
            bitmap.buffer[j] = existBits[j];
        }
        bitmap.density=existBitDensity;
        res->setExistenceBitmap(bitmap);
    }
    std::vector<std::vector<uword>> bitSlices = bringTheBits(nums,slices,attRows);
    for(int i=0; i<slices; i++){
        double bitDensity = bitSlices[i][0]/(double)attRows; // the bit density for this slice
        double compressRatio = 1-pow((1-bitDensity), (2*bits))-pow(bitDensity, (2*bits));
        if(!res->existenceBitmap.isVerbatim()){
            //build compressed bitmap
            HybridBitmap<uword> bitmap;
            HybridBitmapRawIterator<uword> ii = res->existenceBitmap.raw_iterator();
            BufferedRunningLengthWord<uword> &rlwi = ii.next();
            int position = 1;
            while ( rlwi.size() > 0) {
                while (rlwi.getRunningLength() > 0) {
                    bitmap.addStreamOfEmptyWords(0, rlwi.getRunningLength());
                    position += rlwi.getRunningLength();
                    rlwi.discardRunningWordsWithReload();
                }
                const size_t nbre_literal = rlwi.getNumberOfLiteralWords();
                if (nbre_literal > 0) {
                    for (size_t k = 0; k < nbre_literal; ++k) {
                        bitmap.addLiteralWord(bitSlices[i][position]);
                        position++;
                    }
                }
                rlwi.discardLiteralWordsWithReload(nbre_literal);
            }
            bitmap.density=bitDensity;
            bitmap.setSizeInBits(nums.size());
            res->addSlice(bitmap);
            
        }else{
            //build verbatim Bitmap
            HybridBitmap<uword> bitmap(true);
            bitmap.reset();
            bitmap.verbatim = true;
            for (typename std::vector<uword>::iterator it=bitSlices[i].begin()+1; it != bitSlices[i].end(); it++){
                bitmap.addVerbatim(*it,attRows);
            }
            bitmap.setSizeInBits(attRows);
            bitmap.density=bitDensity;
            res->addSlice(bitmap);
            
        }
    }
    res->existenceBitmap.setSizeInBits(attRows,true);
    res->existenceBitmap.density=1;
    res->lastSlice=true;
    res->firstSlice=true;
    res->twosComplement=false;
    res->rows = attRows;
    res->is_signed = true;
    return res;
};

#endif /* BsiAttribute_hpp */
