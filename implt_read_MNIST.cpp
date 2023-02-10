#include "mex.hpp"
#include "mexAdapter.hpp"

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <array>


#define s_usign 4
#define s_char  1

#define magic_number_images 2051
#define magic_number_labels 2049

// This defines the base type, how the data is stored internally
#define BASE_TYPE double

// This prints out an error if an invalid argument is passed, or other errors has happened
#define PRINT_ERR(str){\
    matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({factory.createScalar(str)}));\
}

#define INTCAST(x) static_cast<int>(x)

namespace Types {
    enum Byte {
        UNSIGNED = 4,
        BYTE     = 1
    };

    int to_int (Byte b) {
        return static_cast<int>(b);
    }
};

using mfloat1 = std::vector<double>;
using mfloat2 = std::vector<mfloat1>;

namespace mnist {
using byte = uint8_t;

struct Image {
    int x=0, y=0;
    std::vector<byte> array = {};

    void initalize (int _x, int _y) {
        x = _x; y = _y;
        array.resize(x * y);
    }

    byte &operator() (matlab::data::ArrayFactory &factory, std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr, int _x, int _y=0) {
        int i = _y * y + _x;
        if (i >= x * y)
            PRINT_ERR("mnist::Image: invalid access.");
        return array[i];
    }

    mfloat1 cast () {
        mfloat1 res = {};
        for (size_t i = 0; i < x * y; ++i)
            res.push_back ((double)(array[i]));
        return res;
    }
};

struct Images {
    int magicNumber;
    int nImages;
    int nRows;
    int nCols;
    std::vector<Image> images = {};

    mfloat2 to_floats () {
        mfloat2 res = {};
        for (size_t i = 0; i < nImages; ++i) {
            res.push_back (images[i].cast());
        }
        return res;
    }

    std::vector<Image> as_vectors () {
        std::vector<mnist::Image> res = {};
        for (size_t i = 0; i < nImages; ++i) {
            res.push_back (images[i]);
        }
        return res;
    }

    Images(int mn, int ni, int nr, int nc) : magicNumber(mn), nImages(ni), nRows(nr), nCols(nc) {}
};

struct Labels {
    int magicNumber;
    int nLabels;
    std::vector<byte> labels = {};

    std::vector<double> to_floats () {
        std::vector<double> res = {};
        for (size_t i = 0; i < nLabels; ++i) {
            res.push_back ((double)(labels[i]));
        }
        return res;
    }

    std::vector<byte> as_vectors () {
        std::vector<byte> res = {};
        for (size_t i = 0; i < nLabels; ++i) {
            res.push_back (labels[i]);
        }
        return res;
    }
};
}

unsigned get_data_from_buffer (const std::vector<mnist::byte> &buffer, size_t offset, Types::Byte type, std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr, matlab::data::ArrayFactory &factory) {
    unsigned i_buf = {};

    int until = Types::to_int (type) + offset;
    if (until > buffer.size()) {
        PRINT_ERR("get_data_from_buffer: invalid until.");
    }

    for (int i = offset; i < until; ++i) {
        // store onto i_buf;
        i_buf <<= 8;
        i_buf += (unsigned int)(buffer.at(i));
    }
    return i_buf;
}

mnist::Images implt_get_images (const std::string filename, std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr, matlab::data::ArrayFactory &factory) {
    // read from the file
    FILE *file = std::fopen (filename.c_str(), "rb");
    
    if (file == NULL)
        PRINT_ERR("implt_get_images: file does not exist.");

    // the format is as follows
    /*
     *  offset      what it is          type
     *  0000        magic number        32-bit integer
     *  0004        number of images    32-bit integer
     *  0008        number of rows      32-bit integer
     *  0012        number of columns   32-bit integer
     *  0016        pixel               8-bit  integer
     *  0017        pixel               8-bit  integer
     *  ...
     *  xxxx        pixel               8-bit  integer
     */

    // read to buffer
    std::vector<mnist::byte> g_buffer;
     {
        const size_t max_len = 1 << 10;
        mnist::byte buffer[max_len] = {};

        int len;
        while ((len = std::fread (buffer, sizeof (mnist::byte), max_len, file)) > 0) {
            for (size_t i = 0; i < len; ++i) {
                g_buffer.push_back (buffer[i]);  // read data
                buffer[i] = 0x00;                // clear buffer
            }
        }

     }

     auto mn = get_data_from_buffer      (g_buffer, 0 , Types::UNSIGNED, matlabPtr, factory);
     auto nImages = get_data_from_buffer (g_buffer, 4 , Types::UNSIGNED, matlabPtr, factory);
     auto nRows = get_data_from_buffer   (g_buffer, 8 , Types::UNSIGNED, matlabPtr, factory);
     auto nCols = get_data_from_buffer   (g_buffer, 12, Types::UNSIGNED, matlabPtr, factory);

     if (mn != magic_number_images)
         PRINT_ERR("implt_get_images: input file may be corrupt. magic number does not match.");

    // get the rest of the data
    mnist::Images images {INTCAST(mn),INTCAST(nImages),INTCAST(nRows),INTCAST(nCols)};
    
    images.images.resize (nImages);

    const int nPixels = images.nCols * images.nRows;    // # of pixels per image

    // read in data
    int pixel_count = {};
    int i_image = {};
    for (size_t offset = 16; offset < g_buffer.size(); ++offset) {
        mnist::byte b = get_data_from_buffer (g_buffer, offset, Types::BYTE, matlabPtr, factory);

        if (pixel_count == 0) {
            if (i_image >= images.nImages)
                PRINT_ERR("implt_get_images: data loop: out of bounds access");
            // if the pixel_count is zero, then we need to initalize a new image
            images.images[i_image].initalize(images.nRows, images.nCols);

            // assign byte to image
            images.images[i_image](factory, matlabPtr, pixel_count) = b;
            pixel_count++;

        } 
        else if (pixel_count >= nPixels - 1) {
            images.images[i_image](factory, matlabPtr, pixel_count) = b;

            // if the pixel_count is the number of pixels allocated, increment the image count
            i_image++;

            // reset pixel_count
            pixel_count = 0;

        }
        else {
            // assign to image
            images.images[i_image](factory, matlabPtr, pixel_count) = b;
            pixel_count++;
        }
    }

    if (i_image != nImages)
        PRINT_ERR("implt_get_images: not all images accounted for."); // make sure that the number of available images are accounted for

    std::fclose (file);
    return images;
}

mnist::Labels implt_get_labels (const std::string filename, std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr, matlab::data::ArrayFactory &factory) {
    FILE *file = std::fopen (filename.c_str(), "rb");
    if (file == NULL)
        PRINT_ERR("implt_get_labels: file does not exist.");

    // read into buffer;
    std::vector<mnist::byte> g_buffer;
     {
        const size_t max_len = 1 << 10;
        mnist::byte buffer[max_len] = {};

        int len;
        while ((len = std::fread (buffer, sizeof (mnist::byte), max_len, file)) > 0) {
            for (size_t i = 0; i < len; ++i) {
                g_buffer.push_back (buffer[i]);  // read data
                buffer[i] = 0x00;                // clear buffer
            }
        }

     }
    
    auto mn = get_data_from_buffer      (g_buffer, 0, Types::UNSIGNED, matlabPtr, factory);
    auto nLabels = get_data_from_buffer (g_buffer, 4, Types::UNSIGNED, matlabPtr, factory);
    
    if (mn != magic_number_labels)
        PRINT_ERR("implt_get_labels: file may be corrupt. magic number does not match.");

    mnist::Labels labels;
    labels.magicNumber = INTCAST(mn);
    labels.nLabels     = INTCAST(nLabels);
    labels.labels.resize (nLabels);

    size_t label_count = {};
    for (size_t offset = 8; offset < g_buffer.size(); ++offset) {
        // copy into label buffer
        if (label_count >= nLabels)
            PRINT_ERR("implt_get_labels: too many labels in buffer.");
        labels.labels[label_count] = g_buffer[offset];
        ++label_count;
    }
    
    if (nLabels != label_count)
        PRINT_ERR("implt_get_labels: not all labels accounted for.");

    std::fclose(file);
    return labels;
}

class MexFunction : public matlab::mex::Function  {
    public:
    void operator()(matlab::mex::ArgumentList output, matlab::mex::ArgumentList input) {

        check_arguments(output, input);

        // map Matlab data to c++ standard library strings
        matlab::data::CharArray temp_datafile  = input[0];
        matlab::data::CharArray temp_labelfile = input[1];

        std::string datafile                   = CharArray_To_String(temp_datafile);
        std::string labelfile                  = CharArray_To_String(temp_labelfile);

        combined_data.clear();
        combined_label.clear();

        auto images = implt_get_images (datafile, matlabPtr, factory);
        auto labels = implt_get_labels (labelfile, matlabPtr, factory);

        auto vec_images = images.as_vectors();
        auto vec_labels = labels.as_vectors();

        for (auto & image : vec_images) {
            for (int i = 0; i < image.array.size(); ++i) {
                combined_data.push_back(static_cast<int>(image(factory, matlabPtr, i)));
            }
        }

        for (auto & label : vec_labels) {
            combined_label.push_back(static_cast<int>(label));
        }

        matlab::data::TypedArray<int> v = factory.createArray({static_cast<unsigned long>(images.nRows * images.nCols), static_cast<unsigned long>(images.nImages)}, combined_data.begin() , combined_data.end());

        matlab::data::TypedArray<int> k = factory.createArray({static_cast<unsigned long>(labels.nLabels), 1                                                      }, combined_label.begin(), combined_label.end());

        auto vi = cast_to_vec(images.nImages);
        auto vl = cast_to_vec(labels.nLabels);
        auto vr = cast_to_vec(images.nRows);
        auto vc = cast_to_vec(images.nCols);

        matlab::data::TypedArray<int> di = factory.createArray({1, 1}, vi.begin(), vi.end());
        matlab::data::TypedArray<int> dl = factory.createArray({1, 1}, vl.begin(), vl.end());
        matlab::data::TypedArray<int> dr = factory.createArray({1, 1}, vr.begin(), vr.end());
        matlab::data::TypedArray<int> dc = factory.createArray({1, 1}, vc.begin(), vc.end());

        output[0] = std::move(v);    // return the data
        output[1] = std::move(k);
        output[2] = std::move(di);
        output[3] = std::move(dl);
        output[4] = std::move(dr);
        output[5] = std::move(dc);
    }

    /**
     *  @brief This checks to see if the arguments passed into operator() are valid.
     *  @param
     *   [&output] This is the output argument, a array[] type
     *   [&input]  These are the input arguments, specifically three strings
     *  @throw
     *   feval->error return matlab error when wrong arguments are passed in
     */
    void check_arguments(matlab::mex::ArgumentList &output, matlab::mex::ArgumentList &input) {
        // Check if argument size is correct
        if (input.size() != 2) {
            PRINT_ERR("Two inputs req'd");
        }
    }

    /**
     *  @brief Converts matlab's uint8_t array to a string in C++
     *
     */
    std::string CharArray_To_String(matlab::data::CharArray &arr) {
        std::string buffer;
        for (auto & x : arr)
            buffer += (char)x;
        return buffer;
    }

    std::vector<int> cast_to_vec(int x) {
        std::vector<int> y {x};
        return y;
    }

    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
    matlab::data::ArrayFactory factory;

    std::vector<int> combined_data;
    std::vector<int> combined_label;

};