#include "../bsiCPP/bsi/BsiAttribute.hpp"
#include "../bsiCPP/bsi/hybridBitmap/hybridbitmap.h"
#include <torch/torch.h>

/*std::vector< std::vector< uword > > bringTheBits(BsiAttribute<uword>* build, const torch::Tensor& nums, int slices, int numberOfElements);
BsiAttribute<uword>* buildBsiAttributeFromTensor(BsiAttribute<uword>* build, torch::Tensor& nums, double compressThreshold);
*/
/*
 Building a function to directly build bsi attribute from the PyTorch input tensor
 */

/*template <class uword>
BsiAttribute<uword>* buildBsiAttributeFromTensor(BsiAttribute<uword>* build, torch::Tensor& nums, double compressThreshold) {
    const int MAXLONGLENGTH = 64;
    int slices = 0;
    long min = 0;
    int numberOfElements = nums.numel();
    std::vector<uword> signBits(numberOfElements/(build->bits)+1);
    std::vector<uword> existBits(numberOfElements/(build->bits)+1); // keep track for non-zero values
    int countOnes = 0;
    int CountZeros = 0;
    const uword one = 1;

    for (int i=0; i<numberOfElements; i++){
        int offset = i%(build->bits);
        auto value = nums[i].item<long>(); // Extract a long value from the tensor element
        min = std::min(min, value);
        if(value < 0){
            value = -value;
            signBits[i / (build->bits)] |= (one << offset); // setting sign bit
            countOnes++;
        }
        existBits[i / (build->bits)] |= (one << offset); // setting one at position
        if(value == 0){
            CountZeros++;
        }
        slices = std::max(slices,build->sliceLengthFinder(value)); //Finding the maximum length of the bit representation of the numbers
    }

    BsiAttribute* res;
    if (min < 0) {
        res = new BsiSigned<uword>(slices+1);
        res->sign.reset();
        res->sign.verbatim = true;

        for (typename std::vector<uword>::iterator it=signBits.begin(); it != signBits.end(); it++){
            res->sign.addVerbatim(*it,numberOfElements);
        }
        res->sign.setSizeInBits(numberOfElements);
        res->sign.density = countOnes/(double)numberOfElements;
    } else {
        res = new BsiUnsigned<uword>(slices+1);
    }

    double existBitDensity = (CountZeros/(double)nums.numel()); // to decide whether to compress or not
    double existCompressRatio = 1-pow((1-existBitDensity), (2*build->bits))-pow(existBitDensity, (2*build->bits));
    if(existCompressRatio >= compressThreshold){
        HybridBitmap<uword> bitmap;
        for(int j=0; j<existBits.size(); j++){
            bitmap.addWord(existBits[j]);
        }
        //bitmap.setSizeInBits(numberOfElements);
        bitmap.density=existBitDensity;
        res->setExistenceBitmap(bitmap);
    }else{
        HybridBitmap<uword> bitmap(true,existBits.size());
        for(int j=0; j<existBits.size(); j++){
            bitmap.buffer[j] = existBits[j];
        }
        //bitmap.setSizeInBits(numberOfElements);
        bitmap.density=existBitDensity;
        res->setExistenceBitmap(bitmap);
    }

    //The method to put the elements in the input tensor nums to the bsi property of BSIAttribute result
    std::vector< std::vector< uword > > bitSlices = bringTheBits(nums,slices,numberOfElements);

    for(int i=0; i<slices; i++){
        double bitDensity = bitSlices[i][0]/(double)numberOfElements; // the bit density for this slice
        double compressRatio = 1-pow((1-bitDensity), (2*build->bits))-pow(bitDensity, (2*build->bits));
        if(compressRatio<compressThreshold && compressRatio!=0 ){
            //build compressed bitmap
            HybridBitmap<uword> bitmap;
            for(int j=1; j<bitSlices[i].size(); j++){
                bitmap.addWord(bitSlices[i][j]);
            }
            //bitmap.setSizeInBits(numberOfElements);
            bitmap.density=bitDensity;
            res->addSlice(bitmap);

        }else{
            //build verbatim Bitmap
            HybridBitmap<uword> bitmap(true);
            bitmap.reset();
            bitmap.verbatim = true;
            //                std::copy(bitSlices[i].begin(), bitSlices[i].end(), bitmap.buffer.begin());
            for (typename std::vector<uword>::iterator it=bitSlices[i].begin()+1; it != bitSlices[i].end(); it++){
                bitmap.addVerbatim(*it,numberOfElements);
            }
            // bitmap.buffer=Arrays.copyOfRange(bitSlices[i], 1, bitSlices[i].length);
            //bitmap.actualsizeinwords=bitSlices[i].length-1;
            bitmap.setSizeInBits(numberOfElements);
            bitmap.density=bitDensity;
            res->addSlice(bitmap);

        }
    }
    res->existenceBitmap.setSizeInBits(numberOfElements,true);
    res->existenceBitmap.density=1;
    res->lastSlice=true;
    res->firstSlice=true;
    //res->twosComplement = false;
    res->rows = numberOfElements;
    //res->is_signed = true;
    return res;
};*/
/*
 * Function bringTheBits when input parameter is tensor
 */
/*template <class uword>
std::vector< std::vector< uword > > bringTheBits(BsiAttribute<uword>* build, const torch::Tensor& nums, int slices, int numberOfElements) {
    //The number of words needed to represent the elements in the array
    int wordsNeeded = ceil( numberOfElements / (double)(build->bits));
    //The result of this method is a 2D vector of words
    //Each row represents a slice
    //Each column represents an element in the input array
    //For example, for an input array of 66 ones
    //Two 64bit words are needed to represent 66 numbers
    //Since each element is a one, we need only one slice
    //So the bitmap will be one single row of 66 bits ideally.
    //But in reality, the first word/0th column represents the number of elements.
    //The second word/1st column onwards represents the actual elements
    std::vector< std::vector< uword > > bitmapDataRaw(slices,std::vector<uword>(wordsNeeded +1));
    const uword one = 1;

    // one for the bit density (the first word in each slice)
    uword thisBin = 0;
    for (int seq = 0; seq < numberOfElements; seq++) {
        int w = (seq / (build->bits)+1);
        int offset = seq % (build->bits);
        thisBin = nums[seq].item<long>(); //accessing nums as tensor
        int slice = 0;
        while (thisBin != 0 && slice<slices) {
            if ((thisBin & 1) == 1) {
                bitmapDataRaw[slice][w] |= (one << offset); //setting bit
                bitmapDataRaw[slice][0]++; //update bit density
            }
            thisBin >>= 1;
            slice++;
        }
    }
    return bitmapDataRaw;
};*/