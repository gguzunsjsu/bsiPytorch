//
//  BsiUnsigned.hpp
//

#ifndef BsiUnsigned_cpp
#define BsiUnsigned_cpp

#include <stdio.h>
#include "BsiAttribute.cpp"

template <class uword>
class BsiUnsigned : public BsiAttribute<uword>{
public:
    /*
    Declaring Constructors
     */
    
    BsiUnsigned();
    BsiUnsigned(int maxSize);
    BsiUnsigned(int maxSize, int numOfRows);
    BsiUnsigned(int maxSize, int numOfRows, long partitionID);
    BsiUnsigned(int maxSize, long numOfRows, long partitionID, HybridBitmap<uword> ex);
    
    /*
     Declaring Override Functions
     */
    //template <class uword>
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
    bool append(long value) override;
    
    /*
     * multiplication is only compatible with Verbatim Bitmap
     */
    BsiAttribute<uword>* multiplication(BsiAttribute<uword> *a)const override;
    BsiAttribute<uword>* multiplication_array(BsiAttribute<uword> *a)const override;
    void multiplicationInPlace(BsiAttribute<uword> *a) override;
    void multiply(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const;
    long sumOfBsi()const override;
    HybridBitmap<uword> getExistenceBitmap();
   // void setExistenceBitmap(const HybridBitmap<uword> &exBitmap);
    
    /*
     Declaring Other Functions
     */
    
    BsiAttribute<uword>* SUMunsigned(BsiAttribute<uword>* a)const;
    BsiAttribute<uword>* SUMsigned(BsiAttribute<uword>* a)const;
    BsiAttribute<uword>* SUM(long a, HybridBitmap<uword> EB, int rangeSlices)const;
    
//    BsiAttribute<uword>* negate();
    BsiAttribute<uword>* multiplyWithBSI(BsiUnsigned &unbsi) const;
    BsiAttribute<uword>* multiplyBSI(BsiAttribute<uword> *unbsi)const override;
    BsiUnsigned<uword>& multiplyWithKaratsuba(BsiUnsigned &unbsi) const;
    BsiAttribute<uword>* multiplyWithBsiHorizontal(const BsiAttribute<uword> *unbsi, int precision) const;
    BsiUnsigned<uword>* multiplyBSIWithPrecision(const BsiUnsigned<uword> &unbsi, int precision) const;
    BsiUnsigned<uword>* twosComplement() const;
//    uword sumOfBsi();
    void reset();
    BsiAttribute<uword>* peasantMultiply(BsiUnsigned &unbsi) const;
    int sliceLengthFinder(uword value)const;
    void BitWords(std::vector<uword> &bitWords, long value, int offset);
    
    ~BsiUnsigned();
};



template <class uword>
BsiUnsigned<uword>::~BsiUnsigned(){
    
};

//------------------------------------------------------------------------------------------------------

/*
 Defining Constructors
 */

template <class uword>
BsiUnsigned<uword>::BsiUnsigned() {
    this->size = 0;
    this->bsi.reserve(32);
}

template <class uword>
BsiUnsigned<uword>::BsiUnsigned(int maxSize) {
    this->size = 0;
    this->bsi.reserve(maxSize);
}
/**
 *
 * @param maxSize - maximum number of slices allowed for this attribute
 * @param numOfRows - The number of rows (tuples) in the attribute
 */
template <class uword>
BsiUnsigned<uword>::BsiUnsigned(int maxSize, int numOfRows) {
    this->size = 0;
    this->bsi.reserve(maxSize);
    this->existenceBitmap.setSizeInBits(numOfRows);
    //        if(existenceBitmap.sizeInBits()%64>0)
    //            existenceBitmap.setSizeInBits(existenceBitmap.sizeInBits()+64-existenceBitmap.sizeInBits()%64, false);
    //        existenceBitmap.density = (double)numOfRows/(existenceBitmap.sizeInBits()+64-existenceBitmap.sizeInBits()%64);
    this->existenceBitmap.density=1;
    this->rows = numOfRows;
}

/**
 *
 * @param maxSize - maximum number of slices allowed for this attribute
 * @param numOfRows - The number of rows (tuples) in the attribute
 */
template <class uword>
BsiUnsigned<uword>::BsiUnsigned(int maxSize, int numOfRows, long partitionID) {
    this->size = 0;
    this->bsi.reserve(maxSize);
    this->existenceBitmap.setSizeInBits(numOfRows);
    //        if(existenceBitmap.sizeInBits()%64>0)
    //            existenceBitmap.setSizeInBits(existenceBitmap.sizeInBits()+64-existenceBitmap.sizeInBits()%64, false);
    //        existenceBitmap.density = (double)numOfRows/(existenceBitmap.sizeInBits()+64-existenceBitmap.sizeInBits()%64);
    this->existenceBitmap.density=1;
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
BsiUnsigned<uword>::BsiUnsigned(int maxSize, long numOfRows, long partitionID, HybridBitmap<uword> ex) {
    this->size = 0;
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
 */
template <class uword>
HybridBitmap<uword> BsiUnsigned<uword>::topKMax(int k){
    HybridBitmap<uword> topK, SE, X;
    HybridBitmap<uword> G;
    HybridBitmap<uword> E;
    G.setSizeInBits(this->bsi[0].sizeInBits(),false);
    E.setSizeInBits(this->bsi[0].sizeInBits(),true);
    E.density=1;
    
    int n = 0;
    for (int i = this->size - 1; i >= 0; i--) {
        SE = E.And(this->bsi[i]);
        X = G.Or(SE);
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
    n = G.numberOfOnes();
    topK = G.Or(E);
    // topK = OR(G, E.first(k - n+ 1));
    
    return topK;
};

/**
 * Computes the top-K tuples in a bsi-attribute.
 * @param k - the number in top-k
 * @return a bitArray containing the top-k tuples
 */

template <class uword>
HybridBitmap<uword> BsiUnsigned<uword>::topKMin(int k){
    HybridBitmap<uword> topK, SNOT, X;
    HybridBitmap<uword> G;
    HybridBitmap<uword> E = this->existenceBitmap;
    G.setSizeInBits(this->bsi[0].sizeInBits(),false);
    //E.setSizeInBits(this.bsi[0].sizeInBits(),true);
    //E.density=1;
    int n = 0;
    
    for (int i = this->size - 1; i >= 0; i--) {
        SNOT = E.andNot(this->bsi[i]);
        X = G.Or(SNOT); //Maximum
        n = X.numberOfOnes();
        if (n > k) {
            E = SNOT;
        }
        else if (n < k) {
            G = X;
            E = E.And(this->bsi[i]);
        }
        else {
            E = SNOT;
            break;
        }
    }
    //        n = G.cardinality();
    topK = G.Or(E); //with ties
    // topK = OR(G, E.first(k - n+ 1)); //Exact number of topK
    
    return topK;
};

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::SUM(BsiAttribute<uword>* a)const{
//    HybridBitmap<uword> zeroBitmap;
//    zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
//    BsiAttribute<uword>* res = new BsiUnsigned<uword>(std::max(this->size+this->offset, a->size+a->offset)+1);
//    res->setPartitionID(a->getPartitionID());
//    res->index = this->index;
//    res->existenceBitmap = this->existenceBitmap.Or(a->existenceBitmap);
//    int i = 0, s = a->size, p = this->size;
//
//
//
//    int minOffset = std::min(a->offset,this->offset);
//    res->offset = minOffset;
//
//    int aIndex = 0;
//    int thisIndex =0;
//
//    if(this->offset > a->offset){
//        for(int j=0; j < this->offset-minOffset; j++){
//            if(j<a->size)
//                res->bsi[res->size]=a->bsi[aIndex];
//            else
//                res->bsi[res->size]=zeroBitmap;
//            aIndex++;
//            res->size++;
//        }
//    }else if(a->offset > this->offset){
//        for(int j=0;j<a->offset-minOffset;j++){
//            if(j<this->size)
//                res->bsi[res->size]=this->bsi[thisIndex];
//            else
//                res->bsi[res->size]=zeroBitmap;
//            res->size++;
//            thisIndex++;
//        }
//    }
//    //adjust the remaining sizes for s and p
//    s=s-aIndex;
//    p=p-thisIndex;
//    int minSP = std::min(s, p);
//
//    if(minSP<=0){ // one of the BSI attributes is exausted
//        for(int j=aIndex; j<a->size;j++){
//            res->bsi[res->size]=a->bsi[j];
//            res->size++;
//        }
//        for(int j=thisIndex; j<this->size;j++){
//            res->bsi[res->size]=this->bsi[j];
//            res->size++;
//        }
//        return res;
//    }else {
//
//        res->bsi[res->size] = this->bsi[thisIndex].Xor(a->bsi[aIndex]);
//        HybridBitmap<uword> C = this->bsi[thisIndex].And(a->bsi[aIndex]);
//        res->size++;
//        thisIndex++;
//        aIndex++;
//
//        for(i=1; i<minSP; i++){
//            //res.bsi[i] = this.bsi[i].xor(a.bsi[i].xor(C));
//            res->bsi[res->size] = this->XOR(this->bsi[thisIndex], a->bsi[aIndex], C);
//            //res.bsi[i] = this.bsi[i].xor(this.bsi[i], a.bsi[i], C);
//            C= this->maj(this->bsi[thisIndex], a->bsi[aIndex], C);
//            res->size++;
//            thisIndex++;
//            aIndex++;
//        }
//
//        if(s>p){
//            for(i=p; i<s;i++){
//                res->bsi[res->size] = a->bsi[aIndex].Xor(C);
//                C=a->bsi[aIndex].And(C);
//                res->size++;
//                aIndex++;
//            }
//        }else{
//            for(i=s; i<p;i++){
//                res->bsi[res->size] = this->bsi[thisIndex].Xor(C);
//                C = this->bsi[thisIndex].And(C);
//                res->size++;
//                thisIndex++;
//            }
//        }
//        //if(!(this.lastSlice && a.lastSlice) && (C.cardinality()>0)){
//        if(C.numberOfOnes()>0){
//            res->bsi[res->size]= C;
//            res->size++;
//        }
//        return res;
//    }
    
    if (a->is_signed){
        return SUMsigned(a);
    }else{
        return SUMunsigned(a);
    }
};



template <class uword>
int BsiUnsigned<uword>::sliceLengthFinder(uword value) const{
    //uword mask = 1 << (sizeof(uword) * 8 - 1);
    int lengthCounter =0;
    for(int i = 0; i < sizeof(uword) * 8; i++)
    {
        uword ai = (static_cast<uword>(1) << i);
        if( ( value & (static_cast<uword>(1) << i ) ) != 0 ){
            lengthCounter = i+1;
        }
    }
    return lengthCounter;
}


template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::SUM(long a)const{
        int intSize = sliceLengthFinder(a);
        HybridBitmap<uword> zeroBitmap;
        BsiAttribute<uword>* res;
        zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits(),false);
        HybridBitmap<uword> C;
        if(a<0){
            //int minSP = Math.min(this.size, (intSize+1));
            res = new BsiSigned<uword>(std::max((int)this->size, (intSize+1))+1);
            res->twosComplement=true;
            if ((a&1)==0){
                res->bsi[0]=this->bsi[0];
                C = zeroBitmap;
            }
            else{
                res->bsi[0]=this->bsi[0].logicalnot();
                C=this->bsi[0];
            }
            res->size++;
            int i;
            for( i=1; i<this->size; i++ ){
                if((a&(1<<i))!=0){//xorNot(this->bsi[i])
                    res->bsi[i]=C.xorNot(this->bsi[i]);
                    //res.bsi[i] = C.xor(this.bsi[i].NOT());
                    C=this->bsi[i].Or(C);
                }else{
                    res->bsi[i]=this->bsi[i].Xor(C);
                    C=this->bsi[i].And(C);
                }
                res->size++;
                
            }
            if((intSize+1)>this->size){
                while(i<(intSize+1)){
                    if((a&(1<<i))!=0){
                        res->bsi[i]=C.logicalnot();
                        //C=this.bsi[i].or(C);
                    }else{
                        res->bsi[i]=C;
                        C=zeroBitmap;
                    }
                    i++;
                    res->size++;
                }}else{
                    res->addSlice(C.logicalnot());
                }
            //    if(C.cardinality()!=0){
            //    res.bsi[res.size]=C;
            //res.size++;}
            res->sign = res->bsi[res->size-1];
        }else{
            int minSP = std::min((int)this->size, intSize);
            res = new BsiUnsigned(std::max((int)this->size, intSize)+1);
            HybridBitmap<uword> allOnes;
            allOnes.setSizeInBits(this->bsi[0].sizeInBits(),true);
            allOnes.density=1;
            if ((a&1)==0){
                res->bsi.push_back(this->bsi[0]);
                C = zeroBitmap;
            }
            else{
                res->bsi.push_back(this->bsi[0].logicalnot());
                C=this->bsi[0];
            }
            res->size++;
            int i;
            for(i=1;i<minSP;i++){
                if((a&(1<<i))!=0){
                    res->bsi.push_back(C.xorNot(this->bsi[i]));
                    //res.bsi[i] = C.xor(this.bsi[i].NOT());
                    C=this->bsi[i].Or(C);
                }else{
                    res->bsi.push_back(this->bsi[i].Xor(C));
                    C=this->bsi[i].And(C);
                }
                res->size++;
            }
            long cCard = C.numberOfOnes();
            if(this->size>minSP){
                while(i<this->size){
                    if(cCard>0){
                        res->bsi.push_back(this->bsi[i].Xor(C));
                        C=this->bsi[i].And(C);
                        cCard=C.numberOfOnes();
                    }else{
                        res->bsi.push_back(this->bsi[i]);
                    }
                    res->size++;
                    i++;
                }
            }else{
                while (i<intSize){
                    if(cCard>0){
                        if((a&(1<<i))!=0){
                            res->bsi.push_back(C.logicalnot());
                        }else{
                            res->bsi.push_back(C);
                            C=zeroBitmap;
                            cCard=0;
                        }
                        
                    }else{
                        if((a&(1<<i))!=0){res->bsi[i]=allOnes;
                        }else {res->bsi.push_back(zeroBitmap);}
                        
                    }
                    res->size++;
                    i++;
                }
            }
            if(cCard>0){
                res->bsi.push_back(C);
                res->size++;
            }
            
        }
        res->firstSlice=this->firstSlice;
        res->lastSlice=this->lastSlice;
        res->existenceBitmap = this->existenceBitmap;
        return res;
};

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::convertToTwos(int bitsize){
    BsiSigned<uword>* res = new BsiSigned<uword>();
    res->offset=this->offset;
    res->existenceBitmap = this->existenceBitmap;
    
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.addStreamOfEmptyWords(false,this->existenceBitmap.bufferSize());
    int i=0;
    for(i=0; i<this->getNumberOfSlices(); i++){
        res->addSlice(this->bsi[i]);
    }
    while(i<bitsize){
        res->addSlice(zeroBitmap);
        i++;
    }
    //this.setNumberOfSlices(bits);
    res->setTwosFlag(true);
    
    return res;
};

template <class uword>
long BsiUnsigned<uword>::getValue(int pos){
    long sum = 0;

    for (int i = 0; i < this->size; i++) {
    if(this->bsi[i].get(pos))
        sum += 1l<<(this->offset + i);
    }
    return sum;
};

template <class uword>
HybridBitmap<uword> BsiUnsigned<uword>::rangeBetween(long lowerBound, long upperBound){
    HybridBitmap<uword> B_gt;
    HybridBitmap<uword> B_lt;
    HybridBitmap<uword> B_eq1;
    HybridBitmap<uword> B_eq2;
    HybridBitmap<uword> B_f = this->existenceBitmap;
    B_gt.setSizeInBits(this->bsi[0].sizeInBits());
    B_lt.setSizeInBits(this->bsi[0].sizeInBits());
    B_eq1.setSizeInBits(this->bsi[0].sizeInBits()); B_eq1.density=1;
    B_eq2.setSizeInBits(this->bsi[0].sizeInBits()); B_eq2.density=1;
    
    for(int i=this->getNumberOfSlices()-1; i>=0; i--){
        if((upperBound & (1<<i)) !=0){
            HybridBitmap<uword> ans = B_eq1.andNot(this->bsi[i]);
            //the i'th bit is set in upperBound
            B_lt = B_lt.Or(ans);
            B_eq1 = B_eq1.And(this->bsi[i]);
        }else{ //The i'th bit is not set in uppperBound
            B_eq1=B_eq1.andNot(this->bsi[i]);
        }
        if((lowerBound & (1<<i)) != 0){ // the I'th bit is set in lowerBound
            B_eq2 = B_eq2.And(this->bsi[i]);
        }else{ //the i'th bit is not set in lowerBouond
            B_gt = B_gt.logicalor(B_eq2.And(this->bsi[i]));
            B_eq2 = B_eq2.andNot(this->bsi[i]);
        }
    }
    B_lt = B_lt.Or(B_eq1);
    B_gt = B_gt.Or(B_eq2);
    B_f = B_lt.And(B_gt.And(B_f));
    return B_f;
};

template <class uword>
BsiUnsigned<uword>* BsiUnsigned<uword>::abs(){
    return this;
};

template <class uword>
BsiUnsigned<uword>* BsiUnsigned<uword>::abs(int resultSlices, const HybridBitmap<uword> &EB){
    //    HybridBitmap zeroBitmap = new HybridBitmap();
    //    zeroBitmap.setSizeInBits(this.bsi[0].sizeInBits(),false);
    int min = std::min(this->size-1, resultSlices);
    BsiUnsigned<uword>* res = new BsiUnsigned<uword>(min+1);
    for (int i=0; i<min; i++){
        res->bsi[i]=this->bsi[i];
        res->size++;
    }
    res->size=min;
    res->addSlice(EB.logicalnot()); // this is for KNN to add one slice
    res->existenceBitmap=this->existenceBitmap;
    res->setPartitionID(this->getPartitionID());
    res->firstSlice=this->firstSlice;
    res->lastSlice=this->lastSlice;
    return res;
};

template <class uword>
BsiUnsigned<uword>* BsiUnsigned<uword>::absScale(double range){
    //    HybridBitmap zeroBitmap = new HybridBitmap();
    //    zeroBitmap.setSizeInBits(this.bsi[0].sizeInBits(),false);
    
    HybridBitmap<uword> penalty = this->bsi[this->size-1];
    
    int resSize=0;
    for (int i=this->size-1;i>=0;i--){
        penalty=penalty.logicalor(this->bsi[i]);
        if(penalty.numberOfOnes()>=(this->bsi[0].sizeInBits()*range)){
            //if(penalty.density>=0.9){
            //if(i==this.size-8){
            resSize=i;
            break;
        }
    }
    
    BsiUnsigned<uword> *res = new BsiUnsigned<uword>(resSize+1);
    
    
    
    
    for (int i=0; i<resSize; i++){
        res->bsi[i]=this->bsi[i];
        res->size++;
        
        
    }
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
BsiAttribute<uword>* BsiUnsigned<uword>::SUMunsigned(BsiAttribute<uword>* a)const{
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
    BsiAttribute<uword>* res = new BsiUnsigned<uword>(std::max(this->size+this->offset, a->size+a->offset)+1);
    res->setPartitionID(a->getPartitionID());
    res->existenceBitmap = this->existenceBitmap.logicalor(a->existenceBitmap);
    int i = 0, s = a->size, p = this->size;
    
    
    
    int minOffset = std::min(a->offset, this->offset);
    res->offset = minOffset;
    
    int aIndex = 0;
    int thisIndex =0;
    
    if(this->offset>a->offset){
        for(int j=0;j<this->offset-minOffset; j++){
            if(j<a->size)
                res->bsi[res->size]=a->bsi[aIndex];
            else
                res->bsi[res->size]=zeroBitmap;
            aIndex++;
            res->size++;
        }
    }else if(a->offset>this->offset){
        for(int j=0;j<a->offset-minOffset;j++){
            if(j<this->size)
                res->bsi[res->size]=this->bsi[thisIndex];
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
        for(int j=aIndex; j<a->size;j++){
            res->bsi[res->size]=a->bsi[j];
            res->size++;
        }
        for(int j=thisIndex; j<this->size;j++){
            res->bsi[res->size]=this->bsi[j];
            res->size++;
        }
        return res;
    }else {
        
        res->bsi.push_back(this->bsi[thisIndex].Xor(a->bsi[aIndex]));
        HybridBitmap<uword> C = this->bsi[thisIndex].And(a->bsi[aIndex]);
        res->size++;
        thisIndex++;
        aIndex++;
        
        for(i=1; i<minSP; i++){
            //res.bsi[i] = this.bsi[i].xor(a.bsi[i].xor(C));
            res->bsi.push_back(this->XOR(this->bsi[thisIndex], a->bsi[aIndex], C));
            //res.bsi[i] = this.bsi[i].xor(this.bsi[i], a.bsi[i], C);
            C= this->maj(this->bsi[thisIndex], a->bsi[aIndex], C);
            res->size++;
            thisIndex++;
            aIndex++;
        }
        
        if(s>p){
            for(i=p; i<s;i++){
                res->bsi[res->size] = a->bsi[aIndex].Xor(C);
                C=a->bsi[aIndex].And(C);
                res->size++;
                aIndex++;
            }
        }else{
            for(i=s; i<p;i++){
                res->bsi[res->size] = this->bsi[thisIndex].Xor(C);
                C = this->bsi[thisIndex].And(C);
                res->size++;
                thisIndex++;
            }
        }
        //if(!(this.lastSlice && a.lastSlice) && (C.cardinality()>0)){
        if(C.numberOfOnes()>0){
            res->bsi.push_back( C );
            res->size++;
        }
        return res;
    }
};


template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::SUMsigned(BsiAttribute<uword>* a)const{
    HybridBitmap<uword> zeroBitmap;
    zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
    BsiAttribute<uword>* res = new BsiSigned<uword>(std::max((this->size+this->offset), (a->size+a->offset))+2);
    res->twosComplement=true;
    res->index = (this->index);
    res->existenceBitmap = this->existenceBitmap.logicalor(a->existenceBitmap);
    if (!a->twosComplement)
        a->signMagnitudeToTwos(a->size+1); //plus one for the sign
    
    int i = 0, s = a->size, p = this->size;
    int minOffset = std::min(a->offset, this->offset);
    res->offset = minOffset;
    
    int aIndex = 0;
    int thisIndex =0;
    
    if(this->offset>a->offset){
        for(int j=0;j<this->offset-minOffset; j++){
            if(j<a->size)
                res->bsi[res->size]=a->bsi[aIndex];
            else if(a->lastSlice)
                res->bsi[res->size]=a->sign; //sign extend if contains the sign slice
            else
                res->bsi[res->size] = zeroBitmap;
            aIndex++;
            res->size++;
        }
    }else if(a->offset>this->offset){
        for(int j=0;j<a->offset-minOffset;j++){
            if(j<this->size)
                res->bsi[res->size]=this->bsi[thisIndex];
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
        for(int j=aIndex; j<a->size;j++){
            res->bsi[res->size]=a->bsi[j];
            res->size++;
        }
        HybridBitmap<uword> CC;
        for(int j=thisIndex; j<this->size;j++){
            if(a->lastSlice){ // operate with the sign slice if contains the last slice
                if(j==thisIndex){
                    res->bsi[res->size]=this->bsi[j].logicalxor(a->sign);
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
        
        //res.existenceBitmap = this.existenceBitmap.or(a.existenceBitmap);
        res->sign = &res->bsi[res->size-1];
        res->lastSlice=a->lastSlice;
        res->firstSlice=this->firstSlice|a->firstSlice;
        return res;
    }
    else {
        
        res->bsi[res->size] = this->bsi[thisIndex].logicalxor(a->bsi[aIndex]);
        HybridBitmap<uword> C = this->bsi[thisIndex].logicaland(a->bsi[aIndex]);
        res->size++;
        thisIndex++;
        aIndex++;
        
        for(i=1; i<minSP; i++){
            //res.bsi[i] = this.bsi[i].xor(a.bsi[i].xor(C));
            res->bsi[res->size] =this-> XOR(this->bsi[thisIndex], a->bsi[aIndex], C);
            //res.bsi[i] = this.bsi[i].xor(this.bsi[i], a.bsi[i], C);
            C= this->maj(this->bsi[thisIndex], a->bsi[aIndex], C);
            res->size++;
            thisIndex++;
            aIndex++;
        }
        
        if(s>p){ //a has more bits (the two's complement)
            for(i=p; i<s;i++){
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
            if(this->lastSlice){
                res->bsi[res->size]= C.logicalxor(a->sign);
                C=C.logicaland(a->sign); //
                res->size++;}
        }
        if(!a->lastSlice && C.numberOfOnes()>0){
            //if(!a.lastSlice){
            res->bsi[res->size]= C;
            res->size++;
        }
        
        
        res->lastSlice=a->lastSlice;
        res->firstSlice=this->firstSlice|a->firstSlice;
        res->sign = &res->bsi[res->size-1];
        return res;
    }
};

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::SUM(long a, HybridBitmap<uword> EB, int rangeSlices)const{
    if (a==0){
        return this;
    }else{
        int intSize = (int)std::bitset< 64 >(std::min(std::abs(a),(long)rangeSlices)).to_string().length();
        
        HybridBitmap<uword> zeroBitmap;
        zeroBitmap.setSizeInBits(this->bsi[0].sizeInBits());
        BsiAttribute<uword>* res;
        HybridBitmap<uword> C;
        if(a<0){
            //int minSP = Math.min(this.size, (intSize+1));
            res = new BsiUnsigned<uword>(intSize+1);
            //res.twosComplement=true;
            if ((a&1)==0){
                res->bsi[0]=this->bsi[0].logicaland(EB);
                C = zeroBitmap;
            }
            else{
                res->bsi[0]=EB.logicalandnot(this->bsi[0]);
                C=this->bsi[0].logicaland(EB);
            }
            res->size++;
            int i;
            for( i=1; i<intSize; i++ ){
                if((a&(1<<i))!=0){
                    res->bsi[i]=C.logicalxornot(EB.And(this->bsi[i]));
                    //res.bsi[i] = C.xor(this.bsi[i].NOT());
                    C=EB.logicaland(this->bsi[i]).logicalor(C);
                }else{
                    res->bsi[i]=C.logicalxor(EB.logicaland(this->bsi[i]));
                    //C=this.bsi[i].and(C);
                    C=C.logicaland(EB.logicaland(this->bsi[i].logicaland(C)));
                }
                res->size++;
                
            }
            
            res->addSlice(EB.logicalnot());
            //res.addSlice(C.and(EB));
            //    if(C.cardinality()!=0){
            //    res.bsi[res.size]=C;
            //res.size++;}
            res->sign=&res->bsi[res->size-1];
            res->firstSlice=this->firstSlice;
            res->lastSlice=this->lastSlice;
        }else{
            int minSP = std::min(this->size, intSize);
            res = new BsiUnsigned<uword>(std::max(this->size, intSize)+1);
            //TODO implement this part
        }
        res->existenceBitmap = this->existenceBitmap;
        res->setPartitionID(this->getPartitionID());
        return res;
    }
};

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplyByConstant(int number)const{
    BsiUnsigned<uword>* res = nullptr;
    HybridBitmap<uword> C, S;
    if(number < 0){
        int k = 0;
        number = 0 - number;
        while (number > 0) {
            if ((number & 1) == 1) {
                if (res == nullptr) {
                    res = new BsiUnsigned<uword>();
                    //                res->offset = k;
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
                    // if (A==null || B==null) {
                    // System.out.println("A or B is null");
                    // }
                    S = A.Xor(B);
                    C = A.And(B);
                    // S = XOR_AND(A, B, C);
                    res->bsi[k] = S;
                    
                    // C = Sum[1];
                    
                    for (int i = 1; i < this->size; i++) {// Add the slices of this to the current res
                        
                        B = this->bsi[i];
                        if ((i + k) >=this->size){
                            S = B.Xor(C);
                            C = B.And(C);
                            res->size++;
                            res->bsi.push_back(S);
                            continue;
                            // S = XOR_AND(B, C, C);
                        } else {
                            A = res->bsi[i + k];
                            S = A.Xor(B).Xor(C);
                            // S = XOR(A, B, C);
                            C = A.And(B).Or(B.And(C)).Or(A.And(C));
                            // C = maj(A, B, C); // OR(OR(AND(A, B), AND(A, C)),
                            // AND(C, B));
                        }
                        res->bsi[i + k] = S;
                    }
                    for (int i = this->size + k; i < res->size; i++) {// Add the remaining slices of res with the Carry C
                        A = res->bsi[i];
                        S = A.Xor(C);
                        C = A.And(C);
                        // S = XOR_AND(A, C, C);
                        res->bsi[i] = S;
                    }
                    if (C.numberOfOnes() > 0) {
                        res->bsi.push_back(C); // Carry bit
                        res->size++;
                    }
                    /**/
                }
                // System.out.println("number="+number+" k="+k+" res="+res.SUM());
            }else{
                if (res == nullptr) {
                    res = new BsiUnsigned<uword>();
                    HybridBitmap<uword> zeroBitmap;
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
            //        HybridBitmap<uword> temp;
            //        res->bsi.push_back(temp);
        }
            res->BsiAttribute<uword>::twosComplement = false;
            res->sign.setSizeInBits(this->bsi[0].sizeInBits(), true);
            res->sign.density = 1;
            res->existenceBitmap = this->existenceBitmap;
            res->rows = this->rows;
            res->index = this->index;
        
    }else{
        int k = 0;
        number = 0 - number;
        while (number > 0) {
            if ((number & 1) == 1) {
                if (res == nullptr) {
                    res = new BsiUnsigned<uword>();
                    //                res->offset = k;
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
                    // if (A==null || B==null) {
                    // System.out.println("A or B is null");
                    // }
                    S = A.Xor(B);
                    C = A.And(B);
                    // S = XOR_AND(A, B, C);
                    res->bsi[k] = S;
                    
                    // C = Sum[1];
                    
                    for (int i = 1; i < this->size; i++) {// Add the slices of this to the current res
                        
                        B = this->bsi[i];
                        if ((i + k) >=this->size){
                            S = B.Xor(C);
                            C = B.And(C);
                            res->size++;
                            res->bsi.push_back(S);
                            continue;
                            // S = XOR_AND(B, C, C);
                        } else {
                            A = res->bsi[i + k];
                            S = A.Xor(B).Xor(C);
                            // S = XOR(A, B, C);
                            C = A.And(B).Or(B.And(C)).Or(A.And(C));
                            // C = maj(A, B, C); // OR(OR(AND(A, B), AND(A, C)),
                            // AND(C, B));
                        }
                        res->bsi[i + k] = S;
                    }
                    for (int i = this->size + k; i < res->size; i++) {// Add the remaining slices of res with the Carry C
                        A = res->bsi[i];
                        S = A.Xor(C);
                        C = A.And(C);
                        // S = XOR_AND(A, C, C);
                        res->bsi[i] = S;
                    }
                    if (C.numberOfOnes() > 0) {
                        res->bsi.push_back(C); // Carry bit
                        res->size++;
                    }
                    /**/
                }
                // System.out.println("number="+number+" k="+k+" res="+res.SUM());
            }else{
                if (res == nullptr) {
                    res = new BsiUnsigned<uword>();
                    HybridBitmap<uword> zeroBitmap;
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
            //        HybridBitmap<uword> temp;
            //        res->bsi.push_back(temp);
        }
        
//        res->BsiAttribute<uword>::twosComplement = false;
//        res->sign.setSizeInBits(this->bsi[0].sizeInBits(), true);
        res->sign.density = 1;
        res->existenceBitmap = this->existenceBitmap;
        res->rows = this->rows;
        res->index = this->index;
    }
    
    return res;
};

//template <class uword>
//BsiAttribute<uword>* BsiUnsigned<uword>::negate(){
//    HybridBitmap<uword> onesBitmap;
//    onesBitmap.setSizeInBits(this->bsi[0].sizeInBits(), true);
//    onesBitmap.density=1;
//    
//    int signslicesize=1;
//    if(this->firstSlice)
//        signslicesize=2;
//    
//    BsiSigned<uword>* res = new BsiSigned<uword>(this->getNumberOfSlices()+signslicesize);
//    for(int i=0; i<this->getNumberOfSlices(); i++){
//        res->bsi[i]=this->bsi[i].Not();
//        //            try {
//        //                res.bsi[i]=(HybridBitmap) this.bsi[i].clone();
//        //            } catch (CloneNotSupportedException e) {
//        //                // TODO Auto-generated catch block
//        //                e.printStackTrace();
//        //            }
//        //            res.bsi[i].not();
//        res->size++;
//    }
//    res->addSlice(onesBitmap);
//    
//    if(this->firstSlice){
//        res->addOneSliceNoSignExt(onesBitmap);
//    }
//    res->existenceBitmap=this->existenceBitmap;
//    res->setPartitionID(this->getPartitionID());
//    res->sign=&res->bsi[res->size-1];
//    res->firstSlice=this->firstSlice;
//    res->lastSlice=this->lastSlice;
//    res->setTwosFlag(true);
//    return res;
//};




template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplyWithBSI(BsiUnsigned &unbsi) const{
    BsiUnsigned<uword>* res = nullptr;
    HybridBitmap<uword> C, S, FS, DS;
    int k = 0;
    res = new BsiUnsigned<uword>(this->bsi.size() + unbsi.bsi.size());
    res->offset = k;
    for (int i = 0; i < this->size; i++) {
        res->bsi.push_back(unbsi.bsi[0].andVerbatim(this->bsi[i]));
    }
    res->size = this->size;
    k = 1;
    for (int it=1; it<unbsi.size; it++) {
        /* Move the slices of res k positions */
        HybridBitmap<uword> A, B;
        A = res->bsi[k];
        B = this->bsi[0];
        S = A.xorVerbatim(B);
        C = A.andVerbatim(B);
        FS = unbsi.bsi[it].andVerbatim(S);
        res->bsi[k] = unbsi.bsi[it].selectMultiplication(res->bsi[k],FS);
        //                res->bsi[k] = unbsi.bsi[it].Not().And(res->bsi[k]).Or(unbsi.bsi[it].And(FS));

        for (int i = 1; i < this->size; i++) {// Add the slices of this to the current res
            B = this->bsi[i];
            if ((i + k) < res->size){
                A = res->bsi[i + k];
                S = A.xorVerbatim(B).xorVerbatim(C);
                C = A.maj(B, C);
                //C = A.And(B).Or(B.And(C)).Or(A.And(C));

            } else {
                S = B.xorVerbatim(C);
                C = B.andVerbatim(C);
                res->size++;
                FS = unbsi.bsi[it].andVerbatim(S);
                res->bsi.push_back(FS);
            }
            FS = unbsi.bsi[it].andVerbatim(S);
            res->bsi[i + k] = unbsi.bsi[it].selectMultiplication(res->bsi[i + k],FS);
            //res->bsi[i + k] = res->bsi[i + k].andNot(unbsi.bsi[it]).Or(unbsi.bsi[it].And(FS));
        }
        for (int i = this->size + k; i < res->size; i++) {// Add the remaining slices of res with the Carry C
            A = res->bsi[i];
            S = A.xorVerbatim(C);
            C = A.andVerbatim(C);
            FS = unbsi.bsi[it].andVerbatim(S);
            res->bsi[k] = unbsi.bsi[it].selectMultiplication(res->bsi[k],FS);
        }
        if (C.numberOfOnes() > 0) {
            res->bsi.push_back(unbsi.bsi[it].andVerbatim(C)); // Carry bit
            res->size++;
        }
        k++;
    }
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    return res;
};
//<<<<<<< HEAD
//template <class uword>
//BsiUnsigned<uword>* BsiUnsigned<uword>::twosComplement() const{
//    HybridBitmap<uword> onesBitmap, A,S,C;
//    //C.setSizeInBits(this->bsi[0].sizeInBits(), false);
//    //C.density = 0;
//    onesBitmap.setSizeInBits(this->bsi[0].sizeInBits(), true);
//    onesBitmap.density=1;
//
//    int signslicesize=1;
//    if(this->firstSlice)
//        signslicesize=2;
//
//    BsiUnsigned<uword>* res = new BsiUnsigned<uword>(this->getNumberOfSlices()+signslicesize);
//    A = this->bsi[0].Not();
//    S = A.Xor(onesBitmap);
//    C = A.And(onesBitmap);
//    res->bsi.push_back(S);
//    for(int i=1; i<this->getNumberOfSlices(); i++){
//        A = this->bsi[i].Not();
//        S = A.Xor(C);
//        C = A.And(C);
////        S = A.xorVerbatim(onesBitmap);
////        C = A.andVerbatim(onesBitmap);
//        res->bsi.push_back(S);
//        res->size++;
//    }
//    if (C.numberOfOnes() > 0) {
//        res->bsi.push_back(C); // Carry bit
//        res->size++;
//    }
//    //res->addSlice(onesBitmap);
//
//    if(this->firstSlice){
//        res->addOneSliceNoSignExt(onesBitmap);
//    }
//    res->existenceBitmap=this->existenceBitmap;
//    res->setPartitionID(this->getPartitionID());
//    res->sign=&res->bsi[res->size-1];
//    res->firstSlice=this->firstSlice;
//    res->lastSlice=this->lastSlice;
//    res->setTwosFlag(true);
//    return res;
//};
//
//template <class uword>
//BsiUnsigned<uword>& BsiUnsigned<uword>::multiplyWithKaratsuba(BsiUnsigned &unbsi)const{
//
//=======


template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplyBSI(BsiAttribute<uword> *unbsi) const{
    BsiUnsigned<uword>* res = nullptr;
    HybridBitmap<uword> C, S, FS, DS;
    int k = 0;
    res = new BsiUnsigned<uword>();
    res->offset = k;
    for (int i = 0; i < this->size; i++) {
        res->bsi.push_back(unbsi->bsi[0].And(this->bsi[i]));
    }
    res->size = this->size;
    k = 1;
    for (int it=1; it<unbsi->size; it++) {
        /* Move the slices of res k positions */
        //HybridBitmap<uword> A, B;
        //A = res->bsi[k];
        //B = this->bsi[0];
        S=res->bsi[k];
        S.XorInPlace(this->bsi[0]);
        C = res->bsi[k].And(this->bsi[0]);
        FS = unbsi->bsi[it].And(S);
        //res->bsi[k] = unbsi.bsi[it].selectMultiplication(res->bsi[k],FS);
        res->bsi[k].selectMultiplicationInPlace(unbsi->bsi[it],FS);
//                res->bsi[k] = unbsi.bsi[it].Not().And(res->bsi[k]).Or(unbsi.bsi[it].And(FS));

        for (int i = 1; i < this->size; i++) {// Add the slices of this to the current res
            //B = this->bsi[i];
            if ((i + k) < res->size){
                //A = res->bsi[i + k];
                S=res->bsi[i + k];
                S.XorInPlace(this->bsi[i]);
                S.XorInPlace(C);
                //C = res->bsi[i + k].maj(this->bsi[i], C);
                C.majInPlace(res->bsi[i + k],this->bsi[i]);
                //C = A.And(B).Or(B.And(C)).Or(A.And(C));

            } else {
                S=this->bsi[i];
                S.XorInPlace(C);
                C.AndInPlace(this->bsi[i]);
//                C = this->bsi[i].And(C);
                res->size++;
                FS = unbsi->bsi[it].And(S);
                res->bsi.push_back(FS);
            }
            FS = unbsi->bsi[it].And(S);
            //res->bsi[i + k] = unbsi.bsi[it].selectMultiplication(res->bsi[i + k],FS);
            res->bsi[i+k].selectMultiplicationInPlace(unbsi->bsi[it],FS);
            //res->bsi[i + k] = res->bsi[i + k].andNot(unbsi.bsi[it]).Or(unbsi.bsi[it].And(FS));
        }
        for (int i = this->size + k; i < res->size; i++) {// Add the remaining slices of res with the Carry C
            S = res->bsi[i];
            S.XorInPlace(C);
            C.AndInPlace(res->bsi[i]);
//            C = res->bsi[i].And(C);
            FS = unbsi->bsi[it].And(S);
            //res->bsi[k] = unbsi.bsi[it].selectMultiplication(res->bsi[k],FS);
            res->bsi[k].selectMultiplicationInPlace(unbsi->bsi[it],FS);
        }
        if (C.numberOfOnes() > 0) {
            res->bsi.push_back(unbsi->bsi[it].And(C)); // Carry bit
            res->size++;
        }
        k++;
    }
    
 
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    return res;
};



template <class uword>
BsiUnsigned<uword>* BsiUnsigned<uword>::multiplyBSIWithPrecision(const BsiUnsigned &unbsi, int precision) const{
    
    int precisionInBits = 3*precision +1;
    BsiUnsigned<uword>* res = nullptr;
    HybridBitmap<uword> C, S, FS, DS;
    int k = 0;
    res = new BsiUnsigned<uword>();
    res->offset = k;
    for (int i = 0; i < this->size; i++) {
        res->bsi.push_back(unbsi.bsi[0].andVerbatim(this->bsi[i]));
    }
    res->size = this->size;
    k = 1;
    for (int it=1; it<unbsi.size; it++) {
        /* Move the slices of res k positions */
        //HybridBitmap<uword> A, B;
        //A = res->bsi[k];
        //B = this->bsi[0];
        S=res->bsi[k];
        S.XorInPlace(this->bsi[0]);
        C = res->bsi[k].andVerbatim(this->bsi[0]);
        FS = unbsi.bsi[it].andVerbatim(S);
        //res->bsi[k] = unbsi.bsi[it].selectMultiplication(res->bsi[k],FS);
        res->bsi[k].selectMultiplicationInPlace(unbsi.bsi[it],FS);
        //                res->bsi[k] = unbsi.bsi[it].Not().And(res->bsi[k]).Or(unbsi.bsi[it].And(FS));
        
        for (int i = 1; i < this->size; i++) {// Add the slices of this to the current res
            //B = this->bsi[i];
            if ((i + k) < res->size){
                //A = res->bsi[i + k];
                S=res->bsi[i + k];
                S.XorInPlace(this->bsi[i]);
                S.XorInPlace(C);
                //C = res->bsi[i + k].maj(this->bsi[i], C);
                C.majInPlace(res->bsi[i + k],this->bsi[i]);
                //C = A.And(B).Or(B.And(C)).Or(A.And(C));
                
            } else {
                S=this->bsi[i];
                S.XorInPlace(C);
                C.AndInPlace(this->bsi[i]);
                //                C = this->bsi[i].And(C);
                res->size++;
                FS = unbsi.bsi[it].andVerbatim(S);
                res->bsi.push_back(FS);
            }
            FS = unbsi.bsi[it].andVerbatim(S);
            //res->bsi[i + k] = unbsi.bsi[it].selectMultiplication(res->bsi[i + k],FS);
            res->bsi[i+k].selectMultiplicationInPlace(unbsi.bsi[it],FS);
            //res->bsi[i + k] = res->bsi[i + k].andNot(unbsi.bsi[it]).Or(unbsi.bsi[it].And(FS));
        }
        for (int i = this->size + k; i < res->size; i++) {// Add the remaining slices of res with the Carry C
            S = res->bsi[i];
            S.XorInPlace(C);
            C.AndInPlace(res->bsi[i]);
            //            C = res->bsi[i].And(C);
            FS = unbsi.bsi[it].andVerbatim(S);
            //res->bsi[k] = unbsi.bsi[it].selectMultiplication(res->bsi[k],FS);
            res->bsi[k].selectMultiplicationInPlace(unbsi.bsi[it],FS);
        }
        if (C.numberOfOnes() > 0) {
            res->bsi.push_back(unbsi.bsi[it].andVerbatim(C)); // Carry bit
            res->size++;
        }
        k++;
    }
    
    int truncateBits = res->bsi.size() - precisionInBits;
    for (int i=0; i< truncateBits; i++){
        res->bsi.erase(res->bsi.begin());
    }
    res->size = res->bsi.size();
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    return res;
};


//template <class uword=uint64_t>
//void multiply(uword a[], uword b[], uword ans[], int size_a, int size_b){
//    uword S=0,C=0,FS;
//    int k=0, ansSize=0;
////    uword answer[size_b + size_a];
//    for(int i=0; i<size_a; i++){
//        ans[i] = a[i] & b[0];
//    }
//    for(int i = size_a; i< size_a+size_b; i++){
//        ans[i] = 0;
//    }
//    k=1;
//    ansSize = size_a;
//    for(int it=1; it<size_b; it++){
////        uword A,B;
////        A = answer[k];
////        B = a[0];
//        S = ans[k]^a[0];
//        C = ans[k]&a[0];
//        FS = S & b[it];
//        //~buffer[i] & res.buffer[i]) | (buffer[i] & FS.buffer[i])
//        ans[k] = (~b[it] & ans[k]) | (b[it] & FS);
//
//        for(int i=1; i<size_a; i++){
////            B = a[i];
//            if((i+k) < ansSize){
////                A = answer[i+k];
//                S = ans[i+k] ^ a[i] ^ C;
//                C = (ans[i+k]&a[i]) | (a[i]&C) | (ans[i+k]&C);
//            }else{
//                S = a[i] ^ C;
//                C = a[i] & C;
//                FS = S & b[it];
//                ansSize++;
//                ans[ansSize - 1] = FS;
//            }
//            FS = b[it] & S;
//            ans[i + k] =(~b[it] & ans[i + k]) | (b[it] & FS);
//            //res->bsi[i + k] = res->bsi[i + k].andNot(unbsi.bsi[it]).Or(unbsi.bsi[it].And(FS));
//        }
//        for(int i=size_a + k; i< ansSize; i++){
//            S = ans[i] ^ C;
//            C = ans[i] & C;
//            FS = b[it] & S;
//            ans[k] = (~b[it] & ans[k]) | (b[it] & FS);;
//        }
////        answer[it+k+1] = b[it] & C;
//        if(C>0){
//            ansSize++;
//            ans[ansSize-1] = b[it] & C;
//        }
//        k++;
//    }
////    for(int i= 0; i< size_b + size_a; i++){
////        ans[i] = answer[i];
////    }
//
//};
template <class uword>
void BsiUnsigned<uword>::multiply(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans)const{
    uword S=0,C=0,FS;
    int startPosition = b.size() + a.size() - ans.size();
    int k=0, ansSize=0;
//    uword answer[size_b + size_a];
    
    for(int i=0; i<a.size(); i++){
        ans[i] = a[i] & b[0];
    }
    for(int i = a.size(); i< b.size() + a.size(); i++){
        ans[i] = 0;
    }
    k=1;
    ansSize = a.size();
    for(int it=1; it<b.size(); it++){
        //        uword A,B;
        //        A = answer[k];
        //        B = a[0];
        S = ans[k]^a[0];
        C = ans[k]&a[0];
        FS = S & b[it];
        //~buffer[i] & res.buffer[i]) | (buffer[i] & FS.buffer[i])
        ans[k] = (~b[it] & ans[k]) | (b[it] & FS);
        
        for(int i=1; i<a.size(); i++){
            //            B = a[i];
            if((i+k) < ansSize){
                //                A = answer[i+k];
                S = ans[i+k] ^ a[i] ^ C;
                C = (ans[i+k]&a[i]) | (a[i]&C) | (ans[i+k]&C);
            }else{
                S = a[i] ^ C;
                C = a[i] & C;
                FS = S & b[it];
                ansSize++;
                ans[ansSize - 1] = FS;
            }
            FS = b[it] & S;
            ans[i + k ] =(~b[it] & ans[i + k ]) | (b[it] & FS);
            //res->bsi[i + k] = res->bsi[i + k].andNot(unbsi.bsi[it]).Or(unbsi.bsi[it].And(FS));
        }
        for(int i=a.size() + k; i< ansSize; i++){
            S = ans[i] ^ C;
            C = ans[i] & C;
            FS = b[it] & S;
            ans[k] = (~b[it] & ans[k]) | (b[it] & FS);;
        }
        //        answer[it+k+1] = b[it] & C;
        if(C>0){
            ansSize++;
            ans[ansSize-1] = b[it] & C;
        }
        k++;
    }
//        for(int i= 0; i< size_ans; i++){
//            ans[i] = answer[i];
//        }
    
};


template <class uword=uint64_t>
void multiplyWithPrecision(std::vector<uword> &a, std::vector<uword> &b, std::vector<uword> &ans){
    uword S=0,C=0,FS;
    int k=0, ansSize=0;
    //    uword answer[size_b + size_a];
    for(int i=0; i<a.size(); i++){
        ans[i] = a[i] & b[0];
    }
    for(int i = a.size(); i< ans.size(); i++){
        ans[i] = 0;
    }
    k=1;
    ansSize = a.size();
    for(int it=1; it<b.size(); it++){
        //        uword A,B;
        //        A = answer[k];
        //        B = a[0];
        S = ans[k]^a[0];
        C = ans[k]&a[0];
        FS = S & b[it];
        //~buffer[i] & res.buffer[i]) | (buffer[i] & FS.buffer[i])
        ans[k] = (~b[it] & ans[k]) | (b[it] & FS);
        
        for(int i=1; i<a.size(); i++){
            //            B = a[i];
            if((i+k) < ansSize){
                //                A = answer[i+k];
                S = ans[i+k] ^ a[i] ^ C;
                C = (ans[i+k]&a[i]) | (a[i]&C) | (ans[i+k]&C);
            }else{
                S = a[i] ^ C;
                C = a[i] & C;
                FS = S & b[it];
                ansSize++;
                ans[ansSize - 1] = FS;
            }
            FS = b[it] & S;
            ans[i + k] =(~b[it] & ans[i + k]) | (b[it] & FS);
            //res->bsi[i + k] = res->bsi[i + k].andNot(unbsi.bsi[it]).Or(unbsi.bsi[it].And(FS));
        }
        for(int i=a.size() + k; i< ansSize; i++){
            S = ans[i] ^ C;
            C = ans[i] & C;
            FS = b[it] & S;
            ans[k] = (~b[it] & ans[k]) | (b[it] & FS);;
        }
        //        answer[it+k+1] = b[it] & C;
        if(C>0){
            ansSize++;
            ans[ansSize-1] = b[it] & C;
        }
        k++;
    }
    //    for(int i= 0; i< size_b + size_a; i++){
    //        ans[i] = answer[i];
    //    }
    
};

/*
 */

template <class uword>
BsiAttribute<uword>*  BsiUnsigned<uword>::multiplyWithBsiHorizontal(const BsiAttribute<uword> *unbsi, int precision) const{
    int precisionInBits = 3*precision +1;
    BsiUnsigned<uword>* res = nullptr;
    res = new BsiUnsigned<uword>();
    HybridBitmap<uword> hybridBitmap;
    hybridBitmap.reset();
    hybridBitmap.verbatim = true;
    for(int j=0; j< this->size + unbsi->size; j++){
        res->addSlice(hybridBitmap);
    }
    int size_a = this->size;
    int size_b = unbsi->size;
    std::vector<uword> a(size_a);
    std::vector<uword> b(size_b);
    std::vector<uword> answer(size_a + size_b);
    
    for(int i=0; i< this->bsi[0].bufferSize(); i++){
        for(int j=0; j< this->size; j++){
            a[j] = this->bsi[j].getWord(i); //fetching one word
        }
        for(int j=0; j< unbsi->size; j++){
             b[j] = unbsi->bsi[j].getWord(i);
        }
        this->multiply(a,b,answer);         //perform multiplication on one word
        for(int j=0; j< answer.size() ; j++){
            res->bsi[j].addVerbatim(answer[j]);
        }
    }
    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    return res;
};





template <class uword=uint64_t>
int makeEqualLength(std::vector<uword> &x, std::vector<uword> &y) {
    int len1 = x.size();
    int len2 = y.size();
    if (len1 < len2)
    {
        for (int i = 0 ; i < len2 - len1 ; i++)
            x.push_back((uword)0);
        return len2;
    }
    else if (len1 > len2)
    {
        for (int i = 0 ; i < len1 - len2 ; i++)
            y.push_back((uword)0);
    }
    return len1; // If len1 >= len2
}

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplication(BsiAttribute<uword> *a)const{
    BsiAttribute<uword>* res = multiplyWithBsiHorizontal(a,3);
    int size = res->bsi.size();
    for(int i=0; i<size; i++){
        res->bsi[i].density = res->bsi[i].numberOfOnes()/(double)res->getNumberOfRows();
    }
    return res;
}


template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::multiplication_array(BsiAttribute<uword> *a)const{
    BsiAttribute<uword>* res = multiplyWithBsiHorizontal(a,3);
    int size = res->bsi.size();
    for(int i=0; i<size; i++){
        res->bsi[i].density = res->bsi[i].numberOfOnes()/(double)res->getNumberOfRows();
    }
    return res;
}

template <class uword>
void BsiUnsigned<uword>::multiplicationInPlace(BsiAttribute<uword> *a){
    BsiAttribute<uword>* res = multiplyWithBsiHorizontal(a,3);
    int size = res->bsi.size();
    for(int i=0; i<size; i++){
        res->bsi[i].density = res->bsi[i].numberOfOnes()/(double)res->getNumberOfRows();
    }
   
}
//template <class uword=uint64_t>
//uword multiplyiSingleBit(std::vector<uword> &x, std::vector<uword> &y){
//        uword ans =
//}

template <class uword=uint64_t>
std::vector<uword>& multiplyByKarstuba(std::vector<uword> &x, std::vector<uword> &y, int size_x, int size_y){
    // Find the maximum of lengths of x and Y and make length
    // of smaller string same as that of larger string
    std::vector<uword> answer;
    int n = makeEqualLength(x, y);
    // Base cases
    if (n == 0) return answer.resize(1);
    if (n == 1) return multiplyiSingleBit(x, y);
    
    
}



//
//BsiAttribute<uword>* BsiUnsigned<uword>::karatsuba(BsiUnsigned &a, int startSlice, int endSlice){
//
//
//}
//
///**
// *
// * @tparam uword
// * @param a is the BSI with smaller number of slices
// * @return
// */
//template <class uword>
//BsiAttribute<uword>* BsiUnsigned<uword>::karatsubaMultiply(BsiUnsigned &a){
//    BsiUnsigned<uword>* res = nullptr;
//    HybridBitmap<uword> C; //carry slice
//    long sizeofThis = this->bsi[0].sizeInBits();
//    C.fastaddStreamOfEmptyWords(false,sizeofThis);
//    C.setSizeInBits(sizeofThis);
//    C.density=0;
//
//    //padding a with slices of zeros to make both sides with same number of slices
//    for(int i=a.size; i< this->size; i++){
//        a.addSlice(C);
//    }
//
//    if(this->size==1){
//        return multiplyTwoSlices(this->bsi[0], a.bsi[0]);
//    }
//
//    int firstHalf= this->size/2;
//    BsiAttribute<uword> P1 = this->karatsuba(a,0, firstHalf);
//    BsiAttribute<uword> P2 = this->karatsuba(a,firstHalf+1, this->size);
//    BsiAttribute<uword> P3 = (this.partialSUM(a, 0, firstHalf)).karatsubaMultiply(this.partialSUM(a, firstHalf+1, this->size));
//
//
//
//
//    res->setExistenceBitmap(this->existenceBitmap);
//    res->rows = this->rows;
//    res->index = this->index;
//    return res;
//};


template <class uword>
long BsiUnsigned<uword>::sumOfBsi() const{
    long sum =0;
//    int power = 1;
    for (int i=0; i< this->bsi.size(); i++){
        sum += this->getSlice(i).numberOfOnes()<<(i);
    }
    return sum;
}

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::peasantMultiply(BsiUnsigned &unbsi) const{
    BsiAttribute<uword>* res = nullptr;
    res = new BsiUnsigned<uword>();
    for (int i = 0; i < this->size; i++) {
        res->bsi.push_back(unbsi.bsi[0].And(this->bsi[i]));
    }
    res->size = this->size;
    BsiAttribute<uword> *temp;
    for(int j=1; j<unbsi.size; j++){

        temp = new BsiUnsigned<uword>();
       for (int i = 0; i < this->size; i++) {
           //temp->addSlice(unbsi.bsi[j].And(this->bsi[i]));
            //temp->bsi[i] = unbsi.bsi[j].And(this->bsi[i]);
           temp->bsi.push_back(unbsi.bsi[j].And(this->bsi[i]));
        }
        temp->size = this->size;
        temp->offset=j;
        res=res->SUM(temp);
    }

    res->existenceBitmap = this->existenceBitmap;
    res->rows = this->rows;
    res->index = this->index;
    return res;

};

template <class uword>
void BsiUnsigned<uword>::reset(){
    this->bsi.clear();
    this->size = 0;
    this->rows =0;
    this->index =0;
    int offset =0;
    int decimals = 0;
    
    this->existenceBitmap.reset();
    this->signe = false;
    this->firstSlice = false; //contains first slice
    this->lastSlice = false; //contains last slice

};

template <class uword>
BsiAttribute<uword>* BsiUnsigned<uword>::negate(){
    BsiAttribute<uword>* res = new BsiUnsigned<uword>();
    res->bsi = this->bsi;
    res->sign = new HybridBitmap<uword>(this->getNumberOfRows(),true);
    res->is_signed = true;
    res->twosComplement = false;
    res->setNumberOfRows(this->getNumberOfRows());
    return res;
};


template <class uword>
void BsiUnsigned<uword>::BitWords(std::vector<uword> &bitWords, long value, int offset){
    int i = 0;
    while (value > 0){
        bitWords[i] = (value & 1) << offset;
        value = value/2;
        i++;
    }
}



template <class uword>
bool BsiUnsigned<uword>::append(long value){
    int offset = this->getNumberOfRows()%(sizeof(uword)*8);
    std::vector<uword> bitWords(this->bsi.size());
    BitWords(bitWords, value, offset);
    for (int i=0;i<this->bsi.size(); i++){
        if(this->bsi[i].verbatim == false){
            return false;
        }
    }
    int size = this->bsi[0].buffer.size()-1;
    if(offset == 0){
        for(int i=0; i<this->bsi.size(); i++){
            this->bsi[i].buffer.push_back(bitWords[i]);
            this->bsi[i].setSizeInBits(this->bsi[i].sizeInBits()+1);
        }
    }else{
        for(int i=0; i<this->bsi.size(); i++){
            this->bsi[i].buffer[size] = this->bsi[i].buffer.back() | bitWords[i];
            this->bsi[i].setSizeInBits(this->bsi[i].sizeInBits()+1);
        }
    }
    this->rows++;
    return true;
};

#endif /* BsiUnsigned_hpp */
