//
//  BsiSigned.hpp
//

#ifndef BsiSigned_cpp
#define BsiSigned_cpp

#include <stdio.h>
#include <stdlib.h>     /* abs */
#include "BsiAttribute.cpp"
template <class uword>
class BsiSigned: public BsiAttribute<uword>{
public:
    /*
     Declaring Constructors
     */
    BsiSigned();
    BsiSigned(int maxSize);
    BsiSigned(int maxSize, int numOfRows);
    BsiSigned(int maxSize, int numOfRows, long partitionID);
    BsiSigned(int maxSize, long numOfRows, long partitionID, HybridBitmap<uword> ex);
    
    /*
     Declaring Override Functions
     */
    
    HybridBitmap<uword> topKMax(int k) override;
    HybridBitmap<uword> topKMin(int k) override;
    BsiAttribute<uword>* SUM(BsiAttribute<uword>* a)const override;
    BsiAttribute<uword>* SUM(long a)const override;
    BsiAttribute<uword>* convertToTwos(int bits) override;
    long getValue(int pos) override;
    HybridBitmap<uword> rangeBetween(long lowerBound, long upperBound) override;
    BsiUnsigned<uword>* abs() override;
    BsiUnsigned<uword>* abs(int resultSlices,const HybridBitmap<uword> &EB) override;
    BsiUnsigned<uword>* absScale(double range) override;
    BsiAttribute<uword>* negate() override;
    BsiAttribute<uword>* multiplyByConstant(int number)const override;
    BsiAttribute<uword>* multiplication(BsiAttribute<uword> *a)const override;
    void multiplicationInPlace(BsiAttribute<uword> *a) override;
    long sumOfBsi()const override;
    bool append(long value) override;
    
    /*
     Declaring Other Functions
     */
    void addSliceWithOffset(HybridBitmap<uword> slice, int sliceOffset);
    BsiAttribute<uword>* SUMunsigned(BsiAttribute<uword>* a)const;
    BsiAttribute<uword>* SUMsigned(BsiAttribute<uword>* a)const;
    BsiAttribute<uword>* SUMsignToMagnitude(BsiAttribute<uword>* a)const;
    void twosToSignMagnitude( BsiAttribute<uword>* a)const;
    BsiAttribute<uword>* multiplyWithBsiHorizontal(const BsiAttribute<uword> *a) const;
    void multiply(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans) const;
    void BitWords(std::vector<uword> &bitWords, long value, int offset);
    void appendBitWords(long value);
    void multiply_array(uword a[],int size_a, uword b[], int size_b, uword ans[], int size_ans)const;
    BsiAttribute<uword>* multiplyWithBsiHorizontal_array(const BsiAttribute<uword> *a) const;
    BsiAttribute<uword>* multiplication_Horizontal_Hybrid(const BsiAttribute<uword> *a) const;
    BsiAttribute<uword>* multiplication_Horizontal_Verbatim(const BsiAttribute<uword> *a) const;
    BsiAttribute<uword>* multiplication_Horizontal_compressed(const BsiAttribute<uword> *a) const;
    BsiAttribute<uword>* multiplication_Horizontal_Hybrid_other(const BsiAttribute<uword> *a) const;
    BsiAttribute<uword>* multiplication_Horizontal(const BsiAttribute<uword> *a) const;
    BsiAttribute<uword>* multiplication_array(BsiAttribute<uword> *a)const override;
    BsiAttribute<uword>* multiplyBSI(BsiAttribute<uword> *unbsi)const override;
    
    
    BsiAttribute<uword>* sum_Horizontal_Hybrid(const BsiAttribute<uword> *a) const;
    BsiAttribute<uword>* sum_Horizontal_Verbatim(const BsiAttribute<uword> *a) const;
    BsiAttribute<uword>* sum_Horizontal_compressed(const BsiAttribute<uword> *a) const;
    BsiAttribute<uword>* sum_Horizontal_Hybrid_other(const BsiAttribute<uword> *a) const;
    BsiAttribute<uword>* sum_Horizontal(const BsiAttribute<uword> *a) const;
    void sum(uword a[],int size_a, uword b[], int size_b, uword ans[], int size_ans)const;
    
    
    
    void multiplyKaratsuba(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const;
    void sumOfWordsKaratsuba(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const;
    void subtractionOfWordsKaratsuba(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const;
    void twosComplimentKaratsuba(std::vector<uword> &a)const;
    void combineWordsKaratsuba(std::vector<uword> &a, std::vector<uword> &b,std::vector<uword> &c, std::vector<uword> &d, std::vector<uword> &ac, std::vector<uword> &bd, std::vector<uword> &ans)const;
    void shiftLeftKaratsuba(std::vector<uword> &a, int offset)const;
    void makeEqualLengthKaratsuba(std::vector<uword> &a, std::vector<uword> &b)const;
    void removeZeros(std::vector<uword> &a)const;
    ~BsiSigned();
};



template <class uword>
BsiSigned<uword>::~BsiSigned(){
    
};


//------------------------------------------------------------------------------------------------------

/*
 Defining Constructors
 */

template <class uword>
BsiSigned<uword>::BsiSigned() {
    this->size = 0;
    this->bsi.reserve(BsiAttribute<uword>::bits);
    this->is_signed =true;
}

template <class uword>
BsiSigned<uword>::BsiSigned(int maxSize) {
    this->size = 0;
    this->bsi.reserve(maxSize);
    this->is_signed =true;
}

/**
 *
 * @param maxSize - maximum number of slices allowed for this attribute
 * @param numOfRows - The number of rows (tuples) in the attribute
 */
template <class uword>
BsiSigned<uword>::BsiSigned(int maxSize, int numOfRows) {
    this->size = 0;
    this->signe =true;
    this->bsi.reserve(maxSize);
    this->rows = numOfRows;
}

/**
 *
 * @param maxSize - maximum number of slices allowed for this attribute
 * @param numOfRows - The number of rows (tuples) in the attribute
 */
template <class uword>
BsiSigned<uword>::BsiSigned(int maxSize, int numOfRows, long partitionID) {
    this->size = 0;
    this->signe =true;
    this->bsi.reserve(maxSize);
    this->index=partitionID;
    this->rows = numOfRows;
}

/**
 *
 * @param maxSize - maximum number of slices allowed for this attribute
 * @param numOfRows - The number of rows (tuples) in the attribute
 * @param partitionID - the id of the partition
 * @param ex - existence bitmap
 */

template <class uword>
BsiSigned<uword>::BsiSigned(int maxSize, long numOfRows, long partitionID, HybridBitmap<uword> ex) {
    this->size = 0;
    this->signe =true;
    this->bsi.reserve(maxSize);
    this->existenceBitmap = ex;
    this->index=partitionID;
    
    this->rows = numOfRows;
}


/*
 Defining Override Functions --------------------------------------------------------------------------------------
 */


/**
 * Computes the top-K tuples in a bsi-attribute.
 * @param k - the number in top-k
 * @return a bitArray containing the top-k tuples
 *
 * TokMAx is Compatible with bsi's SignMagnitude Form not Two'sComplement form
 */
template <class uword>
HybridBitmap<uword> BsiSigned<uword>::topKMax(int k){
    
    HybridBitmap<uword> topK, SE, X;
    HybridBitmap<uword> G;
    G.addStreamOfEmptyWords(false, this->existenceBitmap.sizeInBits()/64);
    HybridBitmap<uword> E = this->existenceBitmap.andNot(this->sign); //considers only positive values
    int n = 0;
    for (int i = this->size - 1; i >= 0; i--) {
        SE = E.And(this->bsi[i]);
        X = SE.Or(G);
        n = X.numberOfOnes();
        if (n > k) {
            E = SE;
        }
        if (n < k) {
            G = X;
            E = E.andNot(this->bsi[i]);
        }
        if (n == k) {
            E = SE;
            break;
        }
    }
    if(n<k){
        //todo add negative numbers here (topKMin abs)
    }
    n = G.numberOfOnes();
    topK = G.Or(E);
    return topK;
};

/*
 * topKMin used for find k min values from bsi and return postions bitmap. NOT IMPLEMENTED YET
 */

template <class uword>
HybridBitmap<uword> BsiSigned<uword>::topKMin(int k){

    
    HybridBitmap<uword> h;
    std::cout<<k<<std::endl;
    return h;
};

/*
 * sumOfBsi perform sum vertically on bsi
 */

template <class uword>
long BsiSigned<uword>::sumOfBsi() const{
    long sum =0, minusSum=0;
    //    int power = 1;
    for (int i=0; i< this->getNumberOfSlices(); i++){
        sum += this->getSlice(i).numberOfOnes()<<(i);
    }
    for (int i=0; i< this->getNumberOfSlices(); i++){
        minusSum += this->getSlice(i).And(this->sign).numberOfOnes()<<(i);
    }
    return sum - 2*minusSum;
}


template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::SUM(BsiAttribute<uword>* a) const{
    return sum_Horizontal(a);
    if (a->is_signed and a->twosComplement){
        return this->SUMsignToMagnitude(a);
    }else if(a->is_signed){
        return this->SUMsignToMagnitude(a);
    }
    else{
        return this->SUMunsigned(a);
    }
};


/*
 * add value to every number in BSI
 */


template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::SUM(long a)const{
    
    uword abs_a = std::abs(a);
    int intSize =  BsiAttribute<uword>::sliceLengthFinder(abs_a);
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.addStreamOfEmptyWords(false,this->existenceBitmap.bufferSize());
    BsiAttribute<uword>* res=new BsiSigned<uword>(std::max((int)this->size, intSize)+1);
    
    HybridBitmap<uword> C;
    //int minSP = std::min(this->size, intSize); was not used
    HybridBitmap<uword> allOnes;
    allOnes.setSizeInBits(this->bsi[0].sizeInBits());
    allOnes.density=1;
    if ((a&1)==0){
        res->bsi[0]=this->bsi[0];
        C = zeroBitmap;
    }
    else{
        res->bsi[0]=this->bsi[0].Not();
        C=this->bsi[0];
    }
    res->size++;
    int i;
    for(i=1;i<this->size;i++){
        if((a&(1<<i))!=0){
            res->bsi[i]=C.logicalxornot(this->bsi[i]);
            //res.bsi[i] = C.xor(this.bsi[i].NOT());
            C=this->bsi[i].logicalor(C);
        }else{
            res->bsi[i]=this->bsi[i].logicalxor(C);
            C=this->bsi[i].logicaland(C);
        }
        res->size++;
    }
    if(intSize>this->size){
        while (i<intSize){
            if((a&(1<<i))!=0){
                res->bsi[i]=C.logicalxornot(this->bsi[this->size-1]);
                C=this->bsi[this->size-1].logicalor(C);
            }else{
                res->bsi[i]=C.logicalxor(this->bsi[this->size-1]);
                C=this->bsi[this->size-1].logicaland(C);
            }
            res->size++;
            i++;
        }
    }
    if(this->lastSlice && C.numberOfOnes()>0 ){
        if(a>0){
            res->addSlice(this->sign.logicalandnot(C));
        }else{
            res->addSlice(this->XOR(C,allOnes,this->sign));
        }
    }else{
        res->addSlice(C);
    }
    res->sign = res->bsi[res->size-1];
    res->firstSlice=this->firstSlice;
    res->lastSlice=this->lastSlice;
    res->existenceBitmap = this->existenceBitmap;
    res->twosComplement=false;
    return res;
};


/*
 * convertToTwos converting SignMagnitude to Two'sComplement
 */

template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::convertToTwos(int bits){
    BsiSigned res(bits);
    
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.addStreamOfEmptyWords(false,this->existenceBitmap.bufferSize());
    int i=0;
    for(i=0; i<this->getNumberOfSlices(); i++){
        res.addSlice(this->bsi[i].logicalxor(this->sign));
    }
    while(i<bits){
        res.addSlice(this->sign);
        i++;
    }
    res.addSliceWithOffset(this->sign,0);
    res.setTwosFlag(true);
    
    BsiAttribute<uword>* ans = &res;
    return ans;
};

/*
 * getValue for fetching value of i'th position
 */
template <class uword>
long BsiSigned<uword>::getValue(int i){
    if(this->twosComplement){
        bool sign = this->bsi[this->size-1].get(i);
        long sum=0;
        HybridBitmap<uword> B_i;
        for (int j = 0; j < this->size-1; i++) {
            B_i = this->bsi[j];
            if(B_i.get(i)^sign)
                sum =sum|( 1<<(this->offset + i));
        }
        
        return (sum+((sign)?1:0))*((sign)?-1:1);
    }else{
        long sign = (this->sign.get(i))?-1:1;
        long sum = 0;
        for (int j = 0; j < this->size; j++) {
            if(this->bsi[j].get(i))
                sum += 1<<(this->offset + j);
        }
        return sum*sign;
    }
};

/*
 * Provides values between range in position bitmap: - not implemented yet
 */

template <class uword>
HybridBitmap<uword> BsiSigned<uword>::rangeBetween(long lowerBound, long upperBound){
    //this needs to be implemented
    HybridBitmap<uword> h;
    std::cout<<"lower bound is: "<< lowerBound <<"  uper bound is: "<< upperBound << std::endl;
    return h;
};

/*
 * abs is for converting bsi to two'sComplement to magnitude
 */
template <class uword>
BsiUnsigned<uword>* BsiSigned<uword>::abs(){
    BsiUnsigned<uword>* res = new BsiUnsigned<uword>(this->size);
    if(this->twosComplement){
        for (int i=0; i<this->size-1; i++){
            res->bsi[i]=this->bsi[i].logicalxor(this->sign);
            res->size++;
        }
        if(this->firstSlice){
            res->addOneSliceSameOffset(this->sign);
        }
    }else{
        res->bsi=this->bsi;
        res->size=this->size;
    }
    res->existenceBitmap=this->existenceBitmap;
    res->setPartitionID(this->getPartitionID());
    res->firstSlice=this->firstSlice;
    res->lastSlice=this->lastSlice;
    return res;
};


template <class uword>
BsiUnsigned<uword>* BsiSigned<uword>::abs(int resultSlices,const HybridBitmap<uword> &EB){
    //number of slices allocated for the result; Existence bitmap
    //    HybridBitmap zeroBitmap = new HybridBitmap();
    //    zeroBitmap.setSizeInBits(this.bsi[0].sizeInBits(),false);
    int min = std::min(this->size-1, resultSlices);
    BsiUnsigned<uword>* res = new BsiUnsigned<uword>(min+1);
    
    if(this->twosComplement){
        for (int i=0; i<min; i++){
            res->bsi[i]=this->bsi[i].Xor(this->sign);
        }
        res->size=min;
        if(this->firstSlice){
            res->addOneSliceDiscardCarry(this->sign);
        }
    }else{
        for(int i=0;i<min; i++){
            res->bsi[i]=this->bsi[i];
        }
        res->size=min;
    }
    res->addSlice(EB.Not()); // this is for KNN to add one slice
    res->existenceBitmap=this->existenceBitmap;
    res->setPartitionID(this->getPartitionID());
    res->firstSlice=this->firstSlice;
    res->lastSlice=this->lastSlice;
    return res;
};


template <class uword>
BsiUnsigned<uword>* BsiSigned<uword>::absScale(double range){
    HybridBitmap<uword> penalty = this->bsi[this->size-2].Xor(this->sign);
    
    int resSize=0;
    for (int i=this->size-2;i>=0;i--){
        penalty=penalty.logicalor(this->bsi[i].Xor(this->sign));
        if(penalty.numberOfOnes()>=(this->bsi[0].sizeInBits()*range)){
            resSize=i;
            break;
        }
    }
    
    BsiUnsigned<uword>* res = new BsiUnsigned<uword>(2);
    
    res->addSlice(penalty);
    res->existenceBitmap=this->existenceBitmap;
    res->setPartitionID(this->getPartitionID());
    res->firstSlice=this->firstSlice;
    res->lastSlice=this->lastSlice;
    return res;
};




/*
 Defining Other Functions -----------------------------------------------------------------------------------------
 */



template <class uword>
void BsiSigned<uword>::addSliceWithOffset(HybridBitmap<uword> slice, int sliceOffset){
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
    HybridBitmap<uword> A = this->bsi[sliceOffset-this->offset];
    HybridBitmap<uword> C,S;
    
    S=A.Xor(slice);
    C=A.And(slice);
    
    this->bsi[sliceOffset-this->offset]=S;
    int curPos = sliceOffset-this->offset+1;
    
    while(C.numberOfOnes()>0){
        if(curPos<this->size){
            A=this->bsi[curPos];
            S=C.Xor(A);
            C=C.And(A);
            this->bsi[curPos]=S;
            curPos++;
        }else{
            this->addSlice(C);
        }
    }
}


/*
 * SUMsigned was designed for performing sum without sign bits, which is replaced with
 * SUMsignToMagnitude
 */

template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::SUMunsigned(BsiAttribute<uword>* a)const{
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
    BsiAttribute<uword> *res = new BsiSigned();
    res->twosComplement=true;
    res->setPartitionID(a->getPartitionID());
//    if(!this->twosComplement)
//        this->signMagnitudeToTwos(this->size+1);
//    
    int i = 0, s = a->size, p = this->size, aIndex=0, thisIndex=0;
    int minOffset = std::min(a->offset, this->offset);
    res->offset = minOffset;
    
    if(a->offset>this->offset){
        for(int j=0;j<a->offset-minOffset; j++){
            if(j<this->size)
                res->bsi[res->size]=this->bsi[thisIndex];
            else if(this->lastSlice)
                res->bsi[res->size]=this->sign; //sign extend if contains the sign slice
            else
                res->bsi[res->size]=zeroBitmap;
            thisIndex++;
            res->size++;
        }
    }else if(this->offset>a->offset){
        for(int j=0;j<this->offset-minOffset;j++){
            if(j<a->size)
                res->bsi[res->size]=a->bsi[aIndex];
            else
                res->bsi[res->size]=zeroBitmap;
            res->size++;
            aIndex++;
        }
    }
    //adjust the remaining sizes for s and p
    s=s-aIndex;
    p=p-thisIndex;
    int minSP = std::min(s, p);
    
    if(minSP<=0){ // one of the BSI attributes is exausted
        for(int j=thisIndex; j<this->size;j++){
            res->bsi[res->size]=this->bsi[j];
            res->size++;
        }
        HybridBitmap<uword> CC;
        for(int j=aIndex; j<a->size;j++){
            if(this->lastSlice){ // operate with the sign slice if contains the last slice
                if(j==aIndex){
                    res->bsi[res->size]=a->bsi[j].logicalxor(this->sign);
                    CC=a->bsi[j].logicaland(this->sign);
                    res->lastSlice=true;
                }else{
                    res->bsi[res->size]=this->XOR(a->bsi[j],this->sign,CC);
                    CC=this->maj(a->bsi[j],this->sign,CC);
                }
                res->size++;
            }else{
                res->bsi[res->size]=a->bsi[j];
                res->size++;}
        }
        res->lastSlice=this->lastSlice;
        res->firstSlice=this->firstSlice|a->firstSlice;
        res->existenceBitmap = a->existenceBitmap.logicalor(this->existenceBitmap);
        res->sign = &res->bsi[res->size-1];
        return res;
    }else {
        res->bsi[res->size] = a->bsi[aIndex].logicalxor(this->bsi[thisIndex]);
        HybridBitmap<uword> C = a->bsi[aIndex].logicaland(this->bsi[thisIndex]);
        res->size++;
        thisIndex++;
        aIndex++;
        
        for(i=1; i<minSP; i++){
            //res.bsi[i] = this.bsi[i].xor(a.bsi[i].xor(C));
            res->bsi[res->size] = this->XOR(a->bsi[aIndex], this->bsi[thisIndex], C);
            //res.bsi[i] = this.bsi[i].xor(this.bsi[i], a.bsi[i], C);
            C= this->maj(a->bsi[aIndex], this->bsi[thisIndex], C);
            res->size++;
            thisIndex++;
            aIndex++;
        }
        
        if(s>p){
            for(i=p; i<s;i++){
                res->bsi[res->size] = this->bsi[thisIndex].Xor(C);
                C=this->bsi[thisIndex].logicaland(C);
                res->size++;
                thisIndex++;
            }
        }else{
            for(i=s; i<p;i++){
                if(this->lastSlice){
                    res->bsi[res->size] = this->XOR(a->bsi[aIndex], this->sign, C);
                    C = this->maj(a->bsi[aIndex], this->sign, C);
                    res->size++;
                    aIndex++;}
                else{
                    res->bsi[res->size] = a->bsi[aIndex].Xor(C);
                    C = a->bsi[aIndex].logicaland(C);
                    res->size++;
                    aIndex++;}
            }
        }
        if(!this->lastSlice && C.numberOfOnes()>0){
            res->bsi[res->size]= C;
            res->size++;
        }
        
        res->lastSlice=this->lastSlice;
        res->firstSlice=this->firstSlice|a->firstSlice;
        res->existenceBitmap = a->existenceBitmap.logicalor(this->existenceBitmap);
        res->sign = &res->bsi[res->size-1];
        return res;
    }
};

/*
 * SUMsigned was designed for performing sum with sign bits, which is replaced with SUMsignToMagnitude
 */

template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::SUMsigned( BsiAttribute<uword>* a)const{
    
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
    BsiAttribute<uword>* res = new BsiSigned();
    res->twosComplement=true;
    res->setPartitionID(a->getPartitionID());
    
    if (!a->twosComplement)
        a->signMagnitudeToTwos(a->size+1); //plus one for the sign
    if (!this->twosComplement)
        this->signMagnitudeToTwos(this->size+1); //plus one for the sign
    
    int i = 0, s = a->size, p = this->size, aIndex=0, thisIndex=0;
    int minOffset = std::min(a->offset, this->offset);
    res->offset = minOffset;
    
    if(this->offset>a->offset){
        for(int j=0;j<this->offset-minOffset; j++){
            if(j<a->size)
                res->bsi[res->size]=a->bsi[aIndex];
            else if(a->lastSlice)
                res->bsi[res->size]=a->sign; //sign extend if contains the sign slice
            else
                res->bsi[res->size]=zeroBitmap;
            aIndex++;
            res->size++;
        }
    }else if(a->offset>this->offset){
        for(int j=0;j<a->offset-minOffset;j++){
            if(j<this->size)
                res->bsi[res->size]=this->bsi[thisIndex];
            else if(this->lastSlice)
                res->bsi[res->size]=this->sign;
            else
                res->bsi[res->size]=zeroBitmap;
            res->size++;
            thisIndex++;
        }
    }
    //adjust the remaining sizes for s and p
    s=s-aIndex;
    p=p-thisIndex;
    int minSP = std::min(s, p);
    
    if(minSP<=0){ // one of the BSI attributes is exausted
        HybridBitmap<uword> CC;
        for(int j=aIndex; j<a->size;j++){
            if(this->lastSlice){ // operate with the sign slice if contains the last slice
                if(j==aIndex){
                    res->bsi[res->size]=a->bsi[j].logicalxor(this->sign);
                    CC=a->bsi[j].logicaland(this->sign);
                    res->lastSlice=true;
                }else{
                    res->bsi[res->size]=this->XOR(a->bsi[j],this->sign,CC);
                    CC=this->maj(a->bsi[j],this->sign,CC);
                }
                res->size++;
            }else{
                res->bsi[res->size]=a->bsi[j];
                res->size++;}
        }
        //CC = NULL;
        for(int j=thisIndex; j<this->size;j++){
            if(a->lastSlice){ // operate with the sign slice if contains the last slice
                if(j==thisIndex){
                    res->bsi[res->size]=this->bsi[j].Xor(a->sign);
                    CC=this->bsi[j].logicaland(a->sign);
                    res->lastSlice=true;
                }else{
                    res->bsi[res->size]=this->XOR(this->bsi[j],a->sign,CC);
                    CC=this->maj(this->bsi[j],a->sign,CC);
                }
                res->size++;
            }else{
                res->bsi[res->size]=this->bsi[j];
                res->size++;}
        }
        
        res->lastSlice=this->lastSlice;
        res->firstSlice=this->firstSlice|a->firstSlice;
        res->existenceBitmap = a->existenceBitmap.logicalor(this->existenceBitmap);
        res->sign = &res->bsi[res->size-1];
        return res;
    }else {
        
        res->bsi[res->size] = this->bsi[thisIndex].logicalxor(a->bsi[aIndex]);
        HybridBitmap<uword> C = this->bsi[thisIndex].logicaland(a->bsi[aIndex]);
        res->size++;
        thisIndex++;
        aIndex++;
        
        for(i=1; i<minSP; i++){
            //res.bsi[i] = this.bsi[i].xor(a.bsi[i].xor(C));
            res->bsi[res->size] = this->XOR(this->bsi[thisIndex], a->bsi[aIndex], C);
            //res.bsi[i] = this.bsi[i].xor(this.bsi[i], a.bsi[i], C);
            C= this->maj(this->bsi[thisIndex], a->bsi[aIndex], C);
            res->size++;
            thisIndex++;
            aIndex++;
        }
        
        if(s>p){
            for(i=p; i<s;i++){
                if(this->lastSlice){
                    res->bsi[res->size] = this->XOR(a->bsi[aIndex], this->sign, C);
                    C = this->maj(a->bsi[aIndex], this->sign, C);
                    res->size++;
                    aIndex++;}
                res->bsi[res->size] = a->bsi[aIndex].logicalxor(C);
                C=a->bsi[aIndex].logicaland(C);
                res->size++;
                aIndex++;
            }
        }else{
            for(i=s; i<p;i++){
                if(a->lastSlice){
                    res->bsi[res->size] = this->XOR(this->bsi[thisIndex], a->sign, C);
                    C = this->maj(this->bsi[thisIndex], a->sign, C);
                    res->size++;
                    thisIndex++;}
                else{
                    res->bsi[res->size] = this->bsi[thisIndex].Xor(C);
                    C = this->bsi[thisIndex].logicaland(C);
                    res->size++;
                    thisIndex++;}
            }
        }
        
        if(!this->lastSlice&&!a->lastSlice && C.numberOfOnes()>0){
            res->bsi[res->size]= C;
            res->size++;
        }
        res->sign = this->sign;
        res->existenceBitmap = this->existenceBitmap.logicalor(a->existenceBitmap);
        res->lastSlice=a->lastSlice;
        res->firstSlice=this->firstSlice|a->firstSlice;
        res->setNumberOfRows(this->getNumberOfRows());
        return res;
    }
};



/*
 *  SUMsignToMagnitude takes bsiAttribute as signeMagnitude form perform sumation operation and
 *  return bsiAttribute as signeMagnitude.
 *  signeMagnitude: sign bit is stored separate and only magnitude is stored in bsi.
 */

template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::SUMsignToMagnitude(BsiAttribute<uword>* a) const{
    BsiAttribute<uword> *res = new BsiSigned<uword>();
    if(a->twosComplement or this->twosComplement){
        return res;
    }
    if(this->getNumberOfRows() != a->getNumberOfRows()){
        return res;
    }
    
    int maxSlices = this->getNumberOfSlices() > a->getNumberOfSlices() ? this->getNumberOfSlices() : a->getNumberOfSlices();
    int minSlices = this->getNumberOfSlices() < a->getNumberOfSlices() ? this->getNumberOfSlices() : a->getNumberOfSlices();
    
    HybridBitmap<uword> S;
    HybridBitmap<uword> C = this->sign.xorVerbatim(a->sign);   // Initialize carry with 1 where sign bit is one
    HybridBitmap<uword> signAndBitmap = this->sign.andVerbatim(a->sign);
    HybridBitmap<uword> slice, aSlice,thisSign, aSign;
    
    thisSign = this->sign.xorVerbatim(signAndBitmap);   // Calculating sign bit for Two's compliment
    aSign = a->sign.xorVerbatim(signAndBitmap);
    
    for(int i=0; i<minSlices; i++){
        slice = thisSign.Xor(this->bsi[i]); // converting slice into two's compliment
        aSlice = aSign.Xor(a->bsi[i]);      // converting slice into two's compliment
        S = this->XOR(slice, aSlice, C);
        C = slice.And(aSlice).Or(slice.And(C)).Or(aSlice.And(C));
        res->addSlice(S);
    }
    if(this->getNumberOfSlices() == minSlices){
        for(int i=minSlices; i<maxSlices; i++){
            slice = thisSign;
            aSlice = aSign.Xor(a->bsi[i]);
            S = this->XOR(slice, aSlice, C);
            C = slice.And(aSlice).Or(slice.And(C)).Or(aSlice.And(C));
            res->addSlice(S);
        }
    
    }else{
        for(int i=minSlices; i<maxSlices; i++){
            slice =  thisSign.Xor(this->bsi[i]);
            aSlice = aSign;
            S = this->XOR(slice, aSlice, C);
            C = slice.And(aSlice);
            res->addSlice(S);
        }
    }
    HybridBitmap<uword> signOrBitmap = thisSign.orVerbatim(aSign);
    
//    if(onlyPositiveNumbersCarry.numberOfOnes() >0){
//        res->addSlice(onlyPositiveNumbersCarry);
//    }
    res->addSlice(this->XOR(thisSign, aSign, C));
    res->sign = res->bsi[res->bsi.size()-1].andVerbatim(signOrBitmap);
    res->is_signed = true;
    res->setNumberOfRows(this->getNumberOfRows());
    res->twosComplement = false;
    twosToSignMagnitude(res);   // Converting back to Sign to magnitude form
    res->sign = res->sign.Or(signAndBitmap);
    return res;
};


/*
 *  twosToSignMagnitude is converting Two'sCompliment form into signeMagnitude form
 */
template <class uword>
void  BsiSigned<uword>::twosToSignMagnitude(BsiAttribute<uword>* a) const{
    
    HybridBitmap<uword> C = a->sign;
    HybridBitmap<uword> S,slice;
    for(size_t i=0; i< a->bsi.size(); i++){
        slice = a->sign.Xor(a->bsi[i]);
        S = slice.Xor(C);//a->sign.Xor(a->bsi[i]).Xor(C);
        C = slice.And(C);
        a->bsi[i] = S;
    }
    
//    a->bsi.pop_back();
//    a->setNumberOfSlices(a->bsi.size());
};

/*
 *  negate the sign bit
 */
template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::negate(){
    BsiAttribute<uword>* res = new BsiSigned<uword>();
    res->bsi = this->bsi;
    res->sign = this->sign.Not();
    res->is_signed = true;
    res->twosComplement = false;
    res->setNumberOfRows(this->getNumberOfRows());
    return res;
};


/**
 * Multiplies the BsiAttribute by a constant(Booth's Algorithm)
 * @param number - the constant number
 * @return - the result of the multiplication
 */

template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::multiplyByConstant(int number) const {
    BsiSigned<uword>* res = nullptr;
    HybridBitmap<uword> C, S;
    bool isNegative = false;
    int k = 0;
    if(number < 0){
        isNegative = true;
        number = 0-number;
    }else if(number == 0){
        res = new BsiSigned<uword>();
        HybridBitmap<uword> zeroBitmap;
        zeroBitmap.reset();
        zeroBitmap.verbatim = true;
        int bufferLength = (this->getNumberOfRows()/(this->bits))+1;
        for (int i=0; i<bufferLength; i++){
            zeroBitmap.buffer.push_back(0);
        }
        zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
        res->offset = 0;
        for (int i = 0; i < this->size; i++) {
            res->bsi.push_back(zeroBitmap);
        }
        res->size = this->size;
        res->sign = zeroBitmap;
        res->rows = this->rows;
        res->index = this->index;
        res->is_signed = true;
        res->twosComplement = false;
        res->setNumberOfRows(this->getNumberOfRows());
        return res;
    }
    
    
    while (number > 0) {
        if ((number & 1) == 1) {
            if (res == nullptr) {
                res = new BsiSigned<uword>();
                res->offset = k;
                for (int i = 0; i < this->size; i++) {
                    res->bsi.push_back(this->bsi[i]);
                }
                res->size = this->size;
                k = 0;
            } else {
                /* Move the slices of res k positions */
                HybridBitmap<uword> A, B;
                A = res->bsi[k];
                B = this->bsi[0];
                S = A.Xor(B);
                C = A.And(B);
                res->bsi[k] = S;
                for (int i = 1; i < this->size; i++) {// Add the slices of this to the current res
                    B = this->bsi[i];
                    if ((i + k) >=this->size){
                        S = B.Xor(C);
                        C = B.And(C);
                        res->size++;
                        res->bsi.push_back(S);
                        continue;
                    } else {
                        A = res->bsi[i + k];
                        S = A.Xor(B).Xor(C);
                        C = A.And(B).Or(B.And(C)).Or(A.And(C));
                    }
                    res->bsi[i + k] = S;
                }
                for (int i = this->size + k; i < res->size; i++) {// Add the remaining slices of res with the Carry C
                    A = res->bsi[i];
                    S = A.Xor(C);
                    C = A.And(C);
                    res->bsi[i] = S;
                }
                if (C.numberOfOnes() > 0) {
                    res->bsi.push_back(C); // Carry bit
                    res->size++;
                }
            }
        }else{
            if (res == nullptr) {
                res = new BsiSigned<uword>();
                HybridBitmap<uword> zeroBitmap;
                zeroBitmap.reset();
                zeroBitmap.verbatim = true;
                zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits(), false);
                //                res->offset = k;
                for (int i = 0; i < this->size; i++) {
                    res->bsi.push_back(zeroBitmap);
                }
                res->size = this->size;
                k = 0;
            }
        }
        number >>= 1;
        k++;
    }
    res->existenceBitmap = this->existenceBitmap;
    if(isNegative){
        res->sign = this->sign.Not();
        
    }else{
        res->sign = this->sign;
    }
    res->rows = this->rows;
    res->index = this->index;
    res->is_signed = true;
    res->twosComplement = false;
    res->setNumberOfRows(this->getNumberOfRows());
    return res;
};


/*
 * multiply_array perform multiplication at word level
 * word from every bitmap of Bsi is multiplied with other bsi's word
 * it is modified version of Booth's Algorithm
 */

template <class uword>
void BsiSigned<uword>:: multiply_array(uword a[],int size_a, uword b[], int size_b, uword ans[], int size_ans)const{
    uword S=0,C=0,FS;

    int k=0, ansSize=0;
    for(int i=0; i<size_a; i++){  // Initialing with first bit operation
        ans[i] = a[i] & b[0];
    }
    for(int i = size_a; i< size_a + size_b; i++){ // Initializing rest of bits to zero
        ans[i] = 0;
    }
    k=1;
    ansSize = size_a;
    for(int it=1; it<size_b; it++){
        S = ans[k]^a[0];
        C = ans[k]&a[0];
        FS = S & b[it];
        ans[k] = (~b[it] & ans[k]) | (b[it] & FS); // shifting Operation
        
        for(int i=1; i<size_a; i++){
            int t = i+k;
            if(t < ansSize){
                S = ans[t] ^ a[i] ^ C;
                C = (ans[t]&a[i]) | (a[i]&C) | (ans[t]&C);
            }else{
                S = a[i] ^ C;
                C = a[i] & C;
                FS = S & b[it];
                ansSize++;
                ans[ansSize - 1] = FS;
            }
            FS = b[it] & S;
            ans[i + k ] =(~b[it] & ans[t]) | (b[it] & FS); // shifting Operation
        }
        for(int i=size_a + k; i< ansSize; i++){
            S = ans[i] ^ C;
            C = ans[i] & C;
            FS = b[it] & S;
            ans[k] = (~b[it] & ans[k]) | (b[it] & FS); // shifting Operation
        }
        if(C>0){
            ansSize++;
            ans[ansSize-1] = b[it] & C;
        }
        k++;
    }
    for(int t=ansSize; t<size_ans; t++){
        ans[t] = 0;
    }
};


/*
 * multiplyWithBsiHorizontal_array perform multiplication betwwen bsi using multiply_array
 * only support verbatim Bsi
 */

template <class uword>
BsiAttribute<uword>*  BsiSigned<uword>::multiplyWithBsiHorizontal_array(const BsiAttribute<uword> *a) const{
    //    int precisionInBits = 3*precision + (int)std::log2(precision);
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        res->sign = this->sign.Xor(a->sign);
        res->is_signed = true;
        res->twosComplement = false;
        return res;
    }
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.reset();
    hybridBitmap.verbatim = true;
    for(int j=0; j< this->size + a->size; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->size;
    int size_y = a->size;
    int size_ans = size_y +size_x;
    uword x[size_x];
    uword y[size_y];
    uword answer[size_ans];
    
    for(int i=0; i< this->bsi[0].bufferSize(); i++){
        for(int j=0; j< size_x; j++){
            x[j] = this->bsi[j].getWord(i);
        }
        for(int j=0; j< size_y; j++){
            y[j] = a->bsi[j].getWord(i);
        }
        this->multiply_array(x,size_x,y, size_y,answer, size_ans);
        for(int j=0; j< size_ans ; j++){
            res->bsi[j].addVerbatim(answer[j]);
        }
    }
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
};


/*
 * multiplyWithBsiHorizontal_array perform multiplication betwwen bsi using multiply_array
 * support both verbatim and compressed Bsi(using existenceBitmap)
 */

template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::multiplication_Horizontal(const BsiAttribute<uword> *a) const{
    if(!this->existenceBitmap.isVerbatim() and !a->existenceBitmap.isVerbatim()){
       return this->multiplication_Horizontal_compressed(a);
    }else if (this->existenceBitmap.isVerbatim() or a->existenceBitmap.isVerbatim()){
        if(this->existenceBitmap.verbatim){
            return this->multiplication_Horizontal_Hybrid_other(a);
        }else{
            return this->multiplication_Horizontal_Hybrid(a);
        }
    }else{
        return this->multiplication_Horizontal_Verbatim(a);
    }
    
}

/*
 * multiplication_Horizontal_compressed perform multiplication betwwen bsi using multiply_array
 * only support compressed Bsi(using existenceBitmap)
 */


template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::multiplication_Horizontal_compressed(const BsiAttribute<uword> *a) const{
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        res->sign = this->sign.Xor(a->sign);
        res->is_signed = true;
        res->twosComplement = false;
        return res;
    }
    
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.setSizeInBits(this->existenceBitmap.sizeInBits());
    for(int j=0; j< this->size + a->size; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->size;
    int size_y = a->size;
    int size_ans = size_y +size_x;
    uword x[size_x];
    uword y[size_y];
    uword answer[size_ans];
    
    HybridBitmapRawIterator<uword> iterator = this->existenceBitmap.raw_iterator();
    HybridBitmapRawIterator<uword> a_iterator = a->existenceBitmap.raw_iterator();
    BufferedRunningLengthWord<uword> &rlwi = iterator.next();
    BufferedRunningLengthWord<uword> &rlwa = a_iterator.next();
    
    int position = 0;
    int literal_counter = 1;
    int positionNext = 0;
    int a_literal_counter = 1;
    int a_position = 0;
    int a_positionNext = 0;
    while (rlwi.size() > 0 and rlwa.size() > 0) {
        position = positionNext;
        a_position = a_positionNext;
        while ((rlwi.getRunningLength() > 0) || (rlwa.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwa.getRunningLength();
            BufferedRunningLengthWord<uword> &prey(i_is_prey ? rlwi : rlwa);
            BufferedRunningLengthWord<uword> &predator(i_is_prey ? rlwa : rlwi);
            if (!predator.getRunningBit()) {
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].fastaddStreamOfEmptyWords(false, predator.getRunningLength());
                }
                if(i_is_prey){
                    if(rlwi.getNumberOfLiteralWords() < rlwa.getRunningLength()){
                        position = position+rlwi.getNumberOfLiteralWords()+1;
                        literal_counter =  1;
                    }else{
                        literal_counter += rlwa.getRunningLength() - rlwi.getRunningLength();
                    }
                }else{
                    if(rlwa.getNumberOfLiteralWords() < rlwi.getRunningLength()){
                        a_position = a_position+rlwa.getNumberOfLiteralWords()+1;
                        a_literal_counter = 1;
                    }else{
                        a_literal_counter += rlwi.getRunningLength() - rlwa.getRunningLength();
                    }
                }
                 prey.discardFirstWordsWithReload(predator.getRunningLength());
            }
            predator.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwa.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                for(int j=0; j< size_x; j++){
                    x[j] = this->bsi[j].getWord(position+literal_counter);
                }
                for(int j=0; j< size_y; j++){
                    y[j] = a->bsi[j].getWord(a_position+a_literal_counter);
                }
                this->multiply_array(x,size_x,y, size_y,answer, size_ans);
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].addLiteralWord(answer[j]);
                }
                literal_counter++;
                a_literal_counter++;
            }
            if(rlwi.getNumberOfLiteralWords() == nbre_literal){
                positionNext = position+nbre_literal+1;
                literal_counter = 1;
            }
            if(rlwa.getNumberOfLiteralWords() == nbre_literal){
                a_positionNext = a_position + nbre_literal+1;
                a_literal_counter = 1;
            }
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwa.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
}


/*
 * multiplication_Horizontal_Verbatim perform multiplication betwwen bsi using multiply_array
 * only support verbatim Bsi(using existenceBitmap)
 */


template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::multiplication_Horizontal_Verbatim(const BsiAttribute<uword> *a) const{
    
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        res->sign = this->sign.Xor(a->sign);
        res->is_signed = true;
        res->twosComplement = false;
        return res;
    }
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.reset();
    hybridBitmap.verbatim = true;
    for(int j=0; j< this->size + a->size; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->size;
    int size_y = a->size;
    int size_ans = size_y +size_x;
    uword x[size_x];
    uword y[size_y];
    uword answer[size_ans];
    
    for(size_t i=0; i< this->bsi[0].bufferSize(); i++){
        for(int j=0; j< size_x; j++){
            x[j] = this->bsi[j].getWord(i);
        }
        for(int j=0; j< size_y; j++){
            y[j] = a->bsi[j].getWord(i);
        }
        this->multiply_array(x,size_x,y, size_y,answer, size_ans);
        for(int j=0; j< size_ans ; j++){
            res->bsi[j].addVerbatim(answer[j]);
        }
    }
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
}


/*
 * multiplication_Horizontal_Hybrid perform multiplication betwwen bsi using multiply_array
 * only support hybrid Bsis(one is verbatim and one is compressed)
 */

template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::multiplication_Horizontal_Hybrid(const BsiAttribute<uword> *a) const{

    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        res->sign = this->sign.Xor(a->sign);
        res->is_signed = true;
        res->twosComplement = false;
        return res;
    }

    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.setSizeInBits(this->existenceBitmap.sizeInBits());
    for(int j=0; j< this->size + a->size; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->size;
    int size_y = a->size;
    int size_ans = size_y +size_x;
    uword x[size_x];
    uword y[size_y];
    uword answer[size_ans];
    
    HybridBitmapRawIterator<uword> i = this->existenceBitmap.raw_iterator();
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    
    int positionOfCompressed = 1;
    int positionOfVerbatim = 0;
    while ( rlwi.size() > 0) {
        while (rlwi.getRunningLength() > 0) {
            positionOfVerbatim += rlwi.getRunningLength();
            for(int j=0; j< size_ans ; j++){
                res->bsi[j].addStreamOfEmptyWords(0,rlwi.getRunningLength());
            }
            if(rlwi.getNumberOfLiteralWords() == 0){
                positionOfCompressed++;
            }
            rlwi.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = rlwi.getNumberOfLiteralWords();
        if (nbre_literal > 0) {
            for (size_t k = 1; k <= nbre_literal; ++k) {
                for(int j=0; j< size_x; j++){
                    x[j] = this->bsi[j].getWord(positionOfCompressed +k);
                }
                for(int j=0; j< size_y; j++){
                    y[j] = a->bsi[j].getWord(positionOfVerbatim);
                }
                this->multiply_array(x,size_x,y, size_y,answer, size_ans);
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].addLiteralWord(answer[j]);
                }
                positionOfVerbatim++;
            }
        }
        positionOfCompressed += rlwi.getNumberOfLiteralWords()+1;
        rlwi.discardLiteralWordsWithReload(nbre_literal);
        
    }

    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
};



template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::multiplication_Horizontal_Hybrid_other(const BsiAttribute<uword> *a) const{
    
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        res->sign = this->sign.Xor(a->sign);
        res->is_signed = true;
        res->twosComplement = false;
        return res;
    }
    
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    for(int j=0; j< this->size + a->size; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->size;
    int size_y = a->size;
    int size_ans = size_y +size_x;
    uword x[size_x];
    uword y[size_y];
    uword answer[size_ans];
    
    HybridBitmapRawIterator<uword> i = a->existenceBitmap.raw_iterator();
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    
    int positionOfCompressed = 0;
    int positionOfVerbatim = 0;
    while ( rlwi.size() > 0) {
        while (rlwi.getRunningLength() > 0) {
            positionOfVerbatim += rlwi.getRunningLength();
            for(int j=0; j< size_ans ; j++){
                res->bsi[j].addStreamOfEmptyWords(0,rlwi.getRunningLength());
            }
            if(rlwi.getNumberOfLiteralWords() == 0){
                positionOfCompressed++;
            }
            rlwi.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = rlwi.getNumberOfLiteralWords();
        if (nbre_literal > 0) {
            for (size_t k = 1; k <= nbre_literal; ++k) {
                for(int j=0; j< size_x; j++){
                    x[j] = this->bsi[j].getWord(positionOfVerbatim);
                }
                for(int j=0; j< size_y; j++){
                    y[j] = a->bsi[j].getWord(positionOfCompressed + k);
                }
                this->multiply_array(x,size_x,y, size_y,answer, size_ans);
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].addLiteralWord(answer[j]);
                }
                positionOfVerbatim++;
            }
        }
        positionOfCompressed += rlwi.getNumberOfLiteralWords()+1;
        rlwi.discardLiteralWordsWithReload(nbre_literal);
    }
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
};


/*
 * multiply is modified Booth's algorithm for multiplication design for
 * vertical multiplication of 64-bits at a time same as multiply_array
 */

template <class uword>
void BsiSigned<uword>:: multiply(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const{
    uword S=0,C=0,FS;
    //int startPosition = b.size() + a.size() - ans.size();
    int k=0, ansSize=0;
    for(size_t i=0; i<a.size(); i++){  // Initialing with first bit operation
        ans[i] = a[i] & b[0];
    }
    for(size_t i = a.size(); i< b.size() + a.size(); i++){ // Initializing rest of bits to zero
        ans[i] = 0;
    }
    k=1;
    ansSize = a.size();
    for(size_t it=1; it<b.size(); it++){
        S = ans[k]^a[0];
        C = ans[k]&a[0];
        FS = S & b[it];
        ans[k] = (~b[it] & ans[k]) | (b[it] & FS); // shifting Operation
        
        for(size_t i=1; i<a.size(); i++){
            int t = i+k;
            if(t < ansSize){
                S = ans[t] ^ a[i] ^ C;
                C = (ans[t]&a[i]) | (a[i]&C) | (ans[t]&C);
            }else{
                S = a[i] ^ C;
                C = a[i] & C;
                FS = S & b[it];
                ansSize++;
                ans[ansSize - 1] = FS;
            }
            FS = b[it] & S;
            ans[i + k ] =(~b[it] & ans[t]) | (b[it] & FS); // shifting Operation
        }
        for(int i=a.size() + k; i< ansSize; i++){
            S = ans[i] ^ C;
            C = ans[i] & C;
            FS = b[it] & S;
            ans[k] = (~b[it] & ans[k]) | (b[it] & FS); // shifting Operation
        }
        if(C>0){
            ansSize++;
            ans[ansSize-1] = b[it] & C;
        }
        k++;
    }
};



/*
 * multiplyWithBsiHorizontal perform multiplication betwwen bsi using multiply
 * only support verbatim Bsi
 */


template <class uword>
BsiAttribute<uword>*  BsiSigned<uword>::multiplyWithBsiHorizontal(const BsiAttribute<uword> *a) const{
//    int precisionInBits = 3*precision + (int)std::log2(precision);
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
        res->sign = this->sign.Xor(a->sign);
        res->is_signed = true;
        res->twosComplement = false;
        return res;
    }
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.reset();
    hybridBitmap.verbatim = true;
    for(int j=0; j< this->size + a->size; j++){
        res->addSlice(hybridBitmap);
    }
    int size_x = this->size;
    int size_y = a->size;
    std::vector<uword> x(size_x);
    std::vector<uword> y(size_y);
    std::vector<uword> answer(size_x+size_y);
    
    for(size_t i=0; i< this->bsi[0].bufferSize(); i++){
        for(int j=0; j< size_x; j++){
            x[j] = this->bsi[j].getWord(i);
        }
        for(int j=0; j< size_y; j++){
            y[j] = a->bsi[j].getWord(i);
        }
        
        this->multiply(x,y,answer);
        for(size_t j=0; j< answer.size() ; j++){
            res->bsi[j].addVerbatim(answer[j]);
        }
//        for(int k=answer.size(); k < res->bsi.size(); k++){
//            res->bsi[k].addVerbatim(0);
//        }
        answer.clear();
    }
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed= true;
    res->twosComplement = false;
    return res;
};




/**
 * multiplication the BsiAttribute by another BsiAttribute
 * @param a - the other BsiAttribute
 * @return - the result of the multiplication
 * Only Compatible with verbatim Bitmaps
 */


template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::multiplication(BsiAttribute<uword> *a)const{
    
    BsiAttribute<uword>* res = multiplyWithBsiHorizontal(a);
    int size = res->bsi.size();
    for(int i=0; i< size; i++){
        res->bsi[i].density = res->bsi[i].numberOfOnes()/(double)res->getNumberOfRows();
    }
    return res;
}



template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::multiplication_array(BsiAttribute<uword> *a)const{
    
    BsiAttribute<uword>* res = multiplication_Horizontal(a);
    int size = res->bsi.size();
    for(int i=0; i< size; i++){
        res->bsi[i].density = res->bsi[i].numberOfOnes()/(double)res->getNumberOfRows();
    }
    return res;
}

/*
 * appendBitWords
 */

template <class uword>
void BsiSigned<uword>::appendBitWords(long value){
    
    if (this->bsi.size() == 0){
        int i = 0;
        if(value <0){
            value = std::abs(value);
            this->sign.reset();
            this->sign.verbatim = true;
            this->sign.buffer.push_back((uword)1);
            this->sign.setSizeInBits(1);
        }else{
            this->sign.reset();
            this->sign.verbatim = true;
            this->sign.buffer.push_back((uword)0);
            this->sign.setSizeInBits(1);
        }
        if(value == 0){
            HybridBitmap<uword> zeroBitmap(true,1);
            zeroBitmap.buffer[0] = (0);
            this->bsi.push_back(zeroBitmap);
            this->size++;
        }
        while (value > 0){
            HybridBitmap<uword> zeroBitmap(true,1);
            zeroBitmap.buffer[0] = (value & 1);
            this->bsi.push_back(zeroBitmap);
            this->size++;
            value = value/2;
            i++;
        }
        this->rows = 1;
        this->is_signed = true;
        this->twosComplement =false;

    }else{
        int i = 0;
        int size = this->bsi[0].buffer.size()-1;
        int offset = this->getNumberOfRows()%(BsiAttribute<uword>::bits);
        if(value <0){
            value = std::abs(value);
            this->sign.buffer[size] = this->sign.buffer.back() | (1 << offset);
            this->sign.setSizeInBits(this->sign.sizeInBits()+1);
        }else{
            this->sign.setSizeInBits(this->sign.sizeInBits()+1);
        }
        for(int i=0; i<this->size;i++){
            this->bsi[i].buffer[size] = this->bsi[i].buffer.back() | ((value & 1) << offset);
            this->bsi[i].setSizeInBits(this->bsi[i].sizeInBits()+1);
            value = value/2;
        }
        while (value > 0){
            HybridBitmap<uword> zeroBitmap(true,size+1);
            zeroBitmap.buffer[0] = (value & 1)<< offset;
            zeroBitmap.setSizeInBits(this->rows+1);
            this->bsi.push_back(zeroBitmap);
            value = value/2;
            i++;
            this->size++;
        }
        this->rows++;
       
    }
}

/*
 * Only Use for verbatim bitslices
 */

template <class uword>
bool BsiSigned<uword>::append(long value){
    /*
     * If bitslices are not verbatime compitable
     */

    appendBitWords(value);
    return true;
    }

/*
 * multiplyKaratsuba performs multiplication using Karatsuba algorithm
 * it is lot slower than multiplication_array
 */
template <class uword>
void BsiSigned<uword>::multiplyKaratsuba(std::vector<uword> &A, std::vector<uword> &B, std::vector<uword> &ans)const{
    makeEqualLengthKaratsuba(A,B);
    if(A.size() <= 1){
        for(int i=0; i<A.size(); i++){
            ans.push_back(A[i] & B[i]);
        }
//        multiply(A, B, ans);
    }else{
        int mid = A.size()/2;
        std::vector<uword> b(A.cbegin(),A.cbegin()+mid);
        std::vector<uword> d(B.cbegin(),B.cbegin()+mid);
        std::vector<uword> c(B.cbegin()+mid,B.cend());
        std::vector<uword> a(A.cbegin()+mid,A.cend());
        std::vector<uword> ac(A.size());
        multiply(a, c, ac);
        removeZeros(ac);
        std::vector<uword> bd(B.size());
        multiply(b, d, bd);
        removeZeros(bd);
        combineWordsKaratsuba(a, b, c, d, ac, bd, ans);
        
    }
    
};


/*
 * Calculate Ans = Ac + (a + b)*(c+d)(1<<2*sh) - (ac + bd)(1<<sh) + bd operation.
 */

template <class uword>
void BsiSigned<uword>::combineWordsKaratsuba(std::vector<uword> &a, std::vector<uword> &b,std::vector<uword> &c, std::vector<uword> &d, std::vector<uword> &ac, std::vector<uword> &bd, std::vector<uword> &ans)const{
    std::vector<uword> a_plus_b;
    sumOfWordsKaratsuba(a,b, a_plus_b);
    std::vector<uword> c_plus_d;
    sumOfWordsKaratsuba(c, d, c_plus_d);
    std::vector<uword> a_plus_b_c_plus_d(a_plus_b.size()+c_plus_d.size());
    multiply(a_plus_b, c_plus_d, a_plus_b_c_plus_d);
    removeZeros(a_plus_b_c_plus_d);
    std::vector<uword> ac_plus_bd;
    sumOfWordsKaratsuba(ac, bd, ac_plus_bd);
    makeEqualLengthKaratsuba(a_plus_b_c_plus_d, ac_plus_bd);
    twosComplimentKaratsuba(ac_plus_bd);
    std::vector<uword> middle_word; // (a + b)*(c+d) - (ac + bd)
    subtractionOfWordsKaratsuba(a_plus_b_c_plus_d, ac_plus_bd, middle_word);
    shiftLeftKaratsuba(ac, a.size()+b.size());
    shiftLeftKaratsuba(middle_word, b.size());
    std::vector<uword> firsthalf;
    sumOfWordsKaratsuba(ac, middle_word, firsthalf);
    sumOfWordsKaratsuba(firsthalf, bd, ans);
    removeZeros(ans);
};

template <class uword>
void BsiSigned<uword>::makeEqualLengthKaratsuba(std::vector<uword> &a, std::vector<uword> &b)const{
    if(a.size() > b.size()){
        for (int i=b.size();i<a.size(); i++){
            b.push_back(0);
        }
    }else if (b.size() > a.size()){
        for (int i=a.size();i<b.size(); i++){
            a.push_back(0);
        }
    }
};

template <class uword>
void BsiSigned<uword>::removeZeros(std::vector<uword> &a)const{
    while (a.back() == 0) {
        a.pop_back();
    }
}

template <class uword>
void BsiSigned<uword>::sumOfWordsKaratsuba(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const{
    makeEqualLengthKaratsuba(a,b);
    uword carry = 0;
    for(int i=0; i<a.size(); i++){
        ans.push_back(a[i] ^ b[i] ^ carry);
        carry = (a[i] & b[i]) | (a[i] & carry) | (b[i] & carry);
    }
    if(carry != 0){
        ans.push_back(carry);
    }
};

template <class uword>
void BsiSigned<uword>::subtractionOfWordsKaratsuba(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const{
    uword carry = 0;
    for(int i=0; i<a.size(); i++){
        ans.push_back(a[i] ^ b[i] ^ carry);
        carry = (a[i] & b[i]) | (a[i] & carry) | (b[i] & carry);
    }
};

template <class uword>
void BsiSigned<uword>::twosComplimentKaratsuba(std::vector<uword> &a)const{
    uword carry = ~0;
    for(int i=0; i<a.size(); i++){
        uword ans = ~a[i] ^ carry;
        carry = (~a[i] & carry);
        a[i] = ans;
    }

};

template <class uword>
void BsiSigned<uword>::shiftLeftKaratsuba(std::vector<uword> &a, int offset)const{
    std::vector<uword> ans;
    
    for(int i=0; i<offset; i++){
        ans.push_back(0);
        a.push_back(0);
    }
    for(int i=0; i < a.size() - offset; i++){
        ans.push_back(a[i]);
    }
    for (int i=0; i< ans.size(); i++){
        a[i] = ans[i];
    }
};


/*
 * multiplicationInPlace perfom a *= b using modified booth's algorithm
 */


template <class uword>
void BsiSigned<uword>::multiplicationInPlace(BsiAttribute<uword> *a){
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        return;
    }
    HybridBitmap<uword> hybridBitmap(true,this->bsi[0].buffer.size());
    hybridBitmap.verbatim = true;
    int size = this->size;
    for(int j=size; j< size + a->size; j++){
        this->addSlice(hybridBitmap);
    }
    int size_x = size;
    int size_y = a->size;
    int size_ans = size_y +size_x;
    uword x[size_x];
    uword y[size_y];
    uword answer[size_ans];
    
    for(size_t i=0; i< this->bsi[0].bufferSize(); i++){
        for(int j=0; j< size_x; j++){
            x[j] = this->bsi[j].getWord(i);
        }
        for(int j=0; j< size_y; j++){
            y[j] = a->bsi[j].getWord(i);
        }
        this->multiply_array(x,size_x,y, size_y,answer, size_ans);
        for(int j=0; j< size_ans ; j++){
            this->bsi[j].setWord(i,answer[j]);
        }
    }
    this->size = this->bsi.size();
    for(int i=0; i<this->size; i++){
        this->bsi[i].density = this->bsi[i].numberOfOnes()/(double)this->getNumberOfRows();
    }
    while(this->bsi.back().numberOfOnes() == 0){
        this->bsi.pop_back();
    }
    this->size = this->bsi.size();
    
}



/*
 * multiplicationInPlace perfom a = b * c using modified booth's algorithm
 */


template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::multiplyBSI(BsiAttribute<uword> *a) const{
    BsiAttribute<uword>* res = nullptr;
    HybridBitmap<uword> C, S, FS, DS;
    int k = 0;
    res = new BsiSigned<uword>();
    res->offset = k;
    for (int i = 0; i < this->size; i++) {
        res->bsi.push_back(a->bsi[0].And(this->bsi[i]));
    }
    res->size = this->size;
    k = 1;
    for (int it=1; it<a->size; it++) {
        /* Move the slices of res k positions */
        S=res->bsi[k];
        S = S.Xor(this->bsi[0]);
        C = res->bsi[k].And(this->bsi[0]);
        FS = a->bsi[it].And(S);
        res->bsi[k] = a->bsi[it].Not().And(res->bsi[k]).Or(a->bsi[it].And(FS)); // shifting operation
        
        for (int i = 1; i < this->size; i++) {// Add the slices of this to the current res
            if ((i + k) < res->size){
                //A = res->bsi[i + k];
                S = res->bsi[i + k];
                S = S.Xor(this->bsi[i]);
                S = S.Xor(C);
                C = res->bsi[i + k].And(this->bsi[i]).Or(this->bsi[i].And(C)).Or(res->bsi[i + k].And(C));
                
            } else {
                S=this->bsi[i];
                S = S.Xor(C);
                C = C.And(this->bsi[i]);
                res->size++;
                FS = a->bsi[it].And(S);
                res->bsi.push_back(FS);
            }
            FS = a->bsi[it].And(S);
            res->bsi[i + k] = res->bsi[i + k].andNot(a->bsi[it]).Or(a->bsi[it].And(FS)); // shifting operation
        }
        for (int i = this->size + k; i < res->size; i++) {// Add the remaining slices of res with the Carry C
            S = res->bsi[i];
            S = S.Xor(C);
            C = C.And(res->bsi[i]);
            FS = a->bsi[it].And(S);
            res->bsi[k] = a->bsi[it].Not().And(res->bsi[k]).Or(a->bsi[it].And(FS)); // shifting operation
            
        }
        if (C.numberOfOnes() > 0) {
            res->bsi.push_back(a->bsi[it].And(C)); // Carry bit
            res->size++;
        }
        k++;
    }
    
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
};

/*
 * sum perform sum at word level
 * word from bitmap of one bsi is added to others
 */

template <class uword>
void BsiSigned<uword>::sum(uword a[],int size_a, uword b[], int size_b, uword ans[], int size_ans)const{
    uword carry = 0;
    bool is_a_big = size_a > size_b;
    if(is_a_big){
        for(int i=0; i<size_b; i++){
            ans[i] = (a[i] ^ b[i] ^ carry);
            carry = (a[i] & b[i]) | (a[i] & carry) | (b[i] & carry);
        }
        for(int i=size_b; i<size_a; i++){
            ans[i] = (a[i] ^ carry);
            carry = (a[i] & carry);
        }
            ans[size_ans - 1] = carry;
    }else{
        for(int i=0; i<size_a; i++){
            ans[i] = (a[i] ^ b[i] ^ carry);
            carry = (a[i] & b[i]) | (a[i] & carry) | (b[i] & carry);
        }
        for(int i=size_a; i<size_b; i++){
            ans[i] = (b[i] ^ carry);
            carry = (b[i] & carry);
        }
            ans[size_ans - 1] = carry;
    }
}


/*
 * sum_Horizontal_Hybrid perform summation between two bsi using sum method
 * only support hybrid bsi (one verbatim and one compressed)
 * a is compressed
 */



template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::sum_Horizontal_Hybrid(const BsiAttribute<uword> *a) const{
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        if(this->bsi.size() ==0){
            for(size_t i=0; i<a->bsi.size(); i++){
                res->addSlice(a->bsi[i]);
            }
            res->existenceBitmap = a->existenceBitmap;
            res->rows = a->rows;
            res->index = a->index;
            res->sign = a->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }else{
            for(size_t i=0; i<this->bsi.size(); i++){
                res->addSlice(this->bsi[i]);
            }
            res->existenceBitmap = this->existenceBitmap;
            res->rows = this->rows;
            res->index = this->index;
            res->sign = this->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }
    }
    
    int size_x = this->size;
    int size_y = a->size;
    int size_ans = std::max(size_y,size_x) + 1;
    uword x[size_x];
    uword y[size_y];
    uword answer[size_ans];
    
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap(true,0);
    hybridBitmap.setSizeInBits(a->existenceBitmap.sizeInBits());
    for(int j=0; j< size_ans; j++){
        res->addSlice(hybridBitmap);
    }
    HybridBitmapRawIterator<uword> i = this->existenceBitmap.raw_iterator();
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    
    int positionOfCompressed = 1;
    int positionOfVerbatim = 0;
    while ( rlwi.size() > 0) {
        while (rlwi.getRunningLength() > 0) {
            for(size_t k=0; k<rlwi.getRunningLength(); k++){
                /*
                 * directly adding non zero values to ans if one has zeros
                 */
                for(int j=0; j< size_y; j++){
                    res->bsi[j].addVerbatim(a->bsi[j].buffer[positionOfVerbatim]);
                }
                for(int j=size_y; j< size_ans; j++){
                    res->bsi[j].addVerbatim(0L);
                }
                positionOfVerbatim++;
            }
            
            rlwi.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = rlwi.getNumberOfLiteralWords();
        if (nbre_literal > 0) {
            /*
             * if both have non-zero values
             */
            for (size_t k = 0; k < nbre_literal; ++k) {
                for(int j=0; j< size_x; j++){
                    x[j] = this->bsi[j].getWord(positionOfCompressed);
                }
                for(int j=0; j< size_y; j++){
                    y[j] = a->bsi[j].getWord(positionOfVerbatim);
                }
                this->sum(x,size_x,y, size_y,answer, size_ans);
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].addVerbatim(answer[j]);
                }
                positionOfCompressed++;
                positionOfVerbatim++;
            }
        }
        rlwi.discardLiteralWordsWithReload(nbre_literal);
    }
    
    res->existenceBitmap = a->existenceBitmap;
    res->rows = a->rows;
    res->index = a->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
}


/*
 * sum_Horizontal_Verbatim perform summation between two bsi using sum method
 * only support verbatim bsis
 */

template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::sum_Horizontal_Verbatim(const BsiAttribute<uword> *a) const{

    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        if(this->bsi.size() ==0){
            for(size_t i=0; i<a->bsi.size(); i++){
                res->addSlice(a->bsi[i]);
            }
            res->existenceBitmap = a->existenceBitmap;
            res->rows = a->rows;
            res->index = a->index;
            res->sign = a->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }else{
            for(size_t i=0; i<this->bsi.size(); i++){
                res->addSlice(this->bsi[i]);
            }
            res->existenceBitmap = this->existenceBitmap;
            res->rows = this->rows;
            res->index = this->index;
            res->sign = this->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }
    }
    
    int size_x = this->size;
    int size_y = a->size;
    int size_ans = std::max(size_y,size_x) + 1;
    uword x[size_x];
    uword y[size_y];
    uword answer[size_ans];
    
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap(true,0);
    hybridBitmap.setSizeInBits(a->existenceBitmap.sizeInBits());
    for(int j=0; j< size_ans; j++){
        res->addSlice(hybridBitmap);
    }
    for(size_t i=0; i< this->bsi[0].bufferSize(); i++){
        for(int j=0; j< size_x; j++){
            x[j] = this->bsi[j].getWord(i);
        }
        for(int j=0; j< size_y; j++){
            y[j] = a->bsi[j].getWord(i);
        }
        this->sum(x,size_x,y, size_y,answer, size_ans);
        for(int j=0; j< size_ans ; j++){
            res->bsi[j].addVerbatim(answer[j]);
        }
    }
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
}


/*
 * sum_Horizontal_compressed perform summation between two bsi using sum method
 * only support compressed bsis
 */

template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::sum_Horizontal_compressed(const BsiAttribute<uword> *a) const{
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        if(this->bsi.size() ==0){
            for(size_t i=0; i<a->bsi.size(); i++){
                res->addSlice(a->bsi[i]);
            }
            res->existenceBitmap = a->existenceBitmap;
            res->rows = a->rows;
            res->index = a->index;
            res->sign = a->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }else{
            for(size_t i=0; i<this->bsi.size(); i++){
                res->addSlice(this->bsi[i]);
            }
            res->existenceBitmap = this->existenceBitmap;
            res->rows = this->rows;
            res->index = this->index;
            res->sign = this->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }
    }
    
    int size_x = this->size;
    int size_y = a->size;
    int size_ans = std::max(size_y,size_x) + 1;
    uword x[size_x];
    uword y[size_y];
    uword answer[size_ans];
    
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.setSizeInBits(this->existenceBitmap.sizeInBits());
    for(int j=0; j< size_ans; j++){
        res->addSlice(hybridBitmap);
    }
  
    
    HybridBitmapRawIterator<uword> iterator = this->existenceBitmap.raw_iterator();
    HybridBitmapRawIterator<uword> a_iterator = a->existenceBitmap.raw_iterator();
    BufferedRunningLengthWord<uword> &rlwi = iterator.next();
    BufferedRunningLengthWord<uword> &rlwa = a_iterator.next();
    
   
    
    int position = 0;
    int literal_counter = 1;
    //int positionNext = 0;
    int a_literal_counter = 1;
    int a_position = 0;
    //int a_positionNext = 0;
    while (rlwi.size() > 0 and rlwa.size() > 0) {
        while ((rlwi.getRunningLength() > 0) || (rlwa.getRunningLength() > 0)) {
            const bool i_is_prey = rlwi.getRunningLength() < rlwa.getRunningLength();
            BufferedRunningLengthWord<uword> &prey(i_is_prey ? rlwi : rlwa);
            BufferedRunningLengthWord<uword> &predator(i_is_prey ? rlwa : rlwi);
            if (!prey.getRunningBit() && prey.getRunningLength() >0) {
                // Filling Zeros
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].fastaddStreamOfEmptyWords(false, prey.getRunningLength());
                }
            }
            predator.discardFirstWordsWithReload(prey.getRunningLength());
            prey.discardRunningWordsWithReload();
            if(prey.getNumberOfLiteralWords() >= predator.getRunningLength()){
                for(size_t k=0; k < predator.getRunningLength(); k++){
                    if(i_is_prey){
                        for(int j=0; j< size_x; j++){
                            res->bsi[j].addLiteralWord(this->bsi[j].getWord(position+literal_counter));
                        }
                        for(int j=size_x; j< size_ans; j++){
                            res->bsi[j].addLiteralWord(0);
                        }
                        literal_counter++;
                    }else{
                        for(int j=0; j< size_y; j++){
                            res->bsi[j].addLiteralWord(a->bsi[j].getWord(a_position+a_literal_counter));
                        }
                        for(int j=size_y; j< size_ans; j++){
                            res->bsi[j].addLiteralWord(0);
                        }
                        a_literal_counter++;
                    }
                }
                if(i_is_prey){
                    if(prey.getNumberOfLiteralWords() == predator.getRunningLength()){
                        position +=  literal_counter;
                        literal_counter = 1;
                    }
                }else{
                    if(prey.getNumberOfLiteralWords() == predator.getRunningLength()){
                        a_position +=  a_literal_counter;
                        a_literal_counter = 1;
                    }
                }
                prey.discardFirstWordsWithReload(predator.getRunningLength());
                predator.discardRunningWordsWithReload();
            }else{
                for(size_t k=0; k < prey.getNumberOfLiteralWords(); k++){
                    if(i_is_prey){
                        for(int j=0; j< size_x; j++){
                            res->bsi[j].addLiteralWord(this->bsi[j].getWord(position+literal_counter));
                        }
                        for(int j=size_x; j< size_ans; j++){
                            res->bsi[j].addLiteralWord(0);
                        }
                        literal_counter++;
                    }else{
                        for(int j=0; j< size_y; j++){
                            res->bsi[j].addLiteralWord(a->bsi[j].getWord(a_position+a_literal_counter));
                        }
                        for(int j=size_y; j< size_ans; j++){
                            res->bsi[j].addLiteralWord(0);
                        }
                        a_literal_counter++;
                    }
                }
                
                if(i_is_prey){
                    position += literal_counter;
                    literal_counter = 1;
                }else{
                    a_position += a_literal_counter;
                    a_literal_counter = 1;
                }
                predator.discardFirstWordsWithReload(prey.getNumberOfLiteralWords());
                prey.discardLiteralWordsWithReload(prey.getNumberOfLiteralWords());
            }
        }
        
        const size_t nbre_literal = std::min(rlwi.getNumberOfLiteralWords(),
                                             rlwa.getNumberOfLiteralWords());
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                for(int j=0; j< size_x; j++){
                    x[j] = this->bsi[j].getWord(position+literal_counter);
                }
                for(int j=0; j< size_y; j++){
                    y[j] = a->bsi[j].getWord(a_position+a_literal_counter);
                }
                this->sum(x,size_x,y, size_y,answer, size_ans);
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].addLiteralWord(answer[j]);
                }
                literal_counter++;
                a_literal_counter++;
            }
            if(rlwi.getNumberOfLiteralWords() == nbre_literal){
                position += nbre_literal+1;
                literal_counter = 1;
            }
            if(rlwa.getNumberOfLiteralWords() == nbre_literal){
                a_position += nbre_literal+1;
                a_literal_counter = 1;
            }
            rlwi.discardLiteralWordsWithReload(nbre_literal);
            rlwa.discardLiteralWordsWithReload(nbre_literal);
        }
    }
    
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
}

/*
 * support method for sum_Horizontal_hybrid
 * a is verbatim
 */

template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::sum_Horizontal_Hybrid_other(const BsiAttribute<uword> *a) const{
    if(this->bsi.size() ==0 or a->bsi.size()==0){
        BsiSigned<uword>* res = new BsiSigned<uword>();
        if(this->bsi.size() ==0){
            for(size_t i=0; i<a->bsi.size(); i++){
                res->bsi.push_back(a->bsi[i]);
            }
            res->existenceBitmap = a->existenceBitmap;
            res->rows = a->rows;
            res->index = a->index;
            res->sign = a->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }else{
            for(size_t i=0; i<this->bsi.size(); i++){
                res->bsi.push_back(this->bsi[i]);
            }
            res->existenceBitmap = this->existenceBitmap;
            res->rows = this->rows;
            res->index = this->index;
            res->sign = this->sign;
            res->is_signed = true;
            res->twosComplement = false;
            return res;
        }
    }
    
    int size_x = this->size;
    int size_y = a->size;
    int size_ans = std::max(size_y,size_x) + 1;
    uword x[size_x];
    uword y[size_y];
    uword answer[size_ans];
    
    BsiSigned<uword>* res = new BsiSigned<uword>();
    HybridBitmap<uword> hybridBitmap(true,0);
    hybridBitmap.setSizeInBits(this->existenceBitmap.sizeInBits());
    for(int j=0; j< size_ans; j++){
        res->addSlice(hybridBitmap);
    }
    
    
    HybridBitmapRawIterator<uword> i = a->existenceBitmap.raw_iterator();
    BufferedRunningLengthWord<uword> &rlwi = i.next();
    
    int positionOfCompressed = 1;
    int positionOfVerbatim = 0;
    while ( rlwi.size() > 0) {
        while (rlwi.getRunningLength() > 0) {
            //            positionOfCompressed ++;
            //            positionOfVerbatim += rlwi.getRunningLength();
            for(size_t k=0; k<rlwi.getRunningLength(); k++){
                for(int j=0; j< size_x; j++){
                    res->bsi[j].addVerbatim(this->bsi[j].buffer[positionOfVerbatim]);
                }
                for(int j=size_x; j< size_ans; j++){
                    res->bsi[j].addVerbatim(0L);
                }
                positionOfVerbatim++;
            }
            
            rlwi.discardRunningWordsWithReload();
        }
        const size_t nbre_literal = rlwi.getNumberOfLiteralWords();
        if (nbre_literal > 0) {
            for (size_t k = 0; k < nbre_literal; ++k) {
                //                container.addWord(rlwi.getLiteralWordAt(k));
                for(int j=0; j< size_x; j++){
                    x[j] = this->bsi[j].getWord(positionOfCompressed);
                }
                for(int j=0; j< size_y; j++){
                    y[j] = a->bsi[j].getWord(positionOfVerbatim);
                }
                this->sum(x,size_x,y, size_y,answer, size_ans);
                for(int j=0; j< size_ans ; j++){
                    res->bsi[j].addVerbatim(answer[j]);
                }
                positionOfCompressed++;
                positionOfVerbatim++;
            }
        }
        rlwi.discardLiteralWordsWithReload(nbre_literal);
    }
    
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    res->sign = this->sign.Xor(a->sign);
    res->is_signed = true;
    res->twosComplement = false;
    return res;
}

/*
 * sum_Horizontal perform summation of two bsi using sum method
 */

template <class uword>
BsiAttribute<uword>* BsiSigned<uword>::sum_Horizontal(const BsiAttribute<uword> *a) const{
    if(!this->existenceBitmap.isVerbatim() and !a->existenceBitmap.isVerbatim()){
        return this->sum_Horizontal_compressed(a);
    }else if (this->existenceBitmap.isVerbatim() and a->existenceBitmap.isVerbatim()){
         return this->sum_Horizontal_Verbatim(a);

    }else{
        if(this->existenceBitmap.verbatim){
            return this->sum_Horizontal_Hybrid_other(a);
        }else{
            return this->sum_Horizontal_Hybrid(a);
        }
    }
}
#endif /* BsiSigned_hpp */
