#ifndef UTILSRESNET_H_
#define UTILSRESNET_H_

#include <cublas_v2.h>
#include <vector>
#include <numeric>
#include <string>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <dw/core/Types.h>

// macro to easily check for cuda errors
#define CHECK_CUBLAS_ERROR(x) { \
                    auto result = x; \
                    if(result != CUBLAS_STATUS_SUCCESS) { \
                        char buf[80]; \
                        getDateString(buf, 80); \
                        throw std::runtime_error(std::string(buf) \
                                                + std::string("CUBLAS Error ") \
                                                + std::to_string(result) \
                                                + std::string(" executing CUBLAS function:\n " #x) \
                                                + std::string("\n at " __FILE__ ":") + std::to_string(__LINE__)); \
                    }};




//!
//! \brief  The ManagedCudaBuffer class manages the corresponding device memory allocation and free of buffers.
//!
template<class T>
class ManagedCudaBuffer
{
private:
	ManagedCudaBuffer& operator=(ManagedCudaBuffer& b)
	{
		return b;
	}
	ManagedCudaBuffer(ManagedCudaBuffer& b)
	{
		m_deviceBuffer = nullptr;
		(void)b;
	}

	T* m_deviceBuffer = nullptr;
    size_t m_bufferLen = 0;
public:
	~ManagedCudaBuffer()
	{
		cudaFree(m_deviceBuffer);
	}
	ManagedCudaBuffer(size_t _numElements)
	{
		cudaMalloc(&m_deviceBuffer, _numElements * sizeof(T));
		m_bufferLen = _numElements;
	}
	void copyToHostMemoryAsync(T* hostMemory, cudaStream_t stream)
	{
		cudaMemcpyAsync(hostMemory, m_deviceBuffer, m_bufferLen  * sizeof(T), cudaMemcpyDeviceToHost, stream);
	}

	void copyHostDataToDeviceAsync(T* hostMemory, size_t _numElements, cudaStream_t stream)
	{
		cudaMemcpyAsync(m_deviceBuffer, hostMemory, _numElements* sizeof(T), cudaMemcpyHostToDevice, stream);
	}

	float* getDeviceBuffer() const { return m_deviceBuffer;}
};

//!
//! \brief  read classification labels reference file
//!

inline bool readReferenceFile(const std::string& fileName, std::vector<std::string>& refVector)
{
    std::ifstream infile(fileName);
    if (!infile.is_open())
    {
        std::cout << "ERROR: readReferenceFile: Attempting to read from a file that is not open." << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(infile, line))
    {
        if (line.empty())
            continue;
        refVector.push_back(line);
    }
    infile.close();
    return true;
}


struct PPM
{
    std::string magic, fileName;
    unsigned int h, w, max;
    std::vector<uint8_t> buffer;
};

//!
//! \brief  read a PPM format image.
//!
inline bool readPPMFile(const std::string& filename, PPM& ppm)
{
	bool validImage = true;
    ppm.fileName = filename;
    std::ifstream infile(filename, std::ifstream::binary);
    if (infile.is_open())
    {
		infile >> ppm.magic ;
		if(ppm.magic == "P6")
		{
			// skip comment lines
			std::string line;
			getline(infile, line);
			while((line.size() == 0) || (line[0] == '#'))
			{
				getline(infile, line);
			}
			// continue reading description
			infile.seekg(-(line.size()+1), infile.cur);
			infile >> ppm.w >> ppm.h >> ppm.max;
			// read image data
			getline(infile, line);
			while((line.size() == 0))
			{
				getline(infile, line);
			}
			infile.seekg(-(line.size()+1), infile.cur);
			ppm.buffer.resize(ppm.w * ppm.h * 3);
			infile.read(reinterpret_cast<char*>(ppm.buffer.data()), ppm.w * ppm.h * 3);
		}
		else
		{
			validImage = false;
		}
		infile.close();
    }
    else
    {
    	validImage = false;
    }
    return validImage;

}

//!
//! \brief  write a PPM format image.
//!
inline void writePPMFile(const std::string& filename, PPM& ppm)
{
    std::ofstream outfile("./" + filename, std::ofstream::binary);
    assert(!outfile.fail());
    outfile << "P6"
            << "\n"
            << ppm.w << " " << ppm.h << "\n"
            << ppm.max << "\n";
    outfile.write(reinterpret_cast<const char*>(ppm.buffer.data()), ppm.w * ppm.h * 3);
}


// Locate path to file, given its filename or filepath suffix and possible dirs it might lie in
// Function will also walk back MAX_DEPTH dirs from CWD to check for such a file path
inline std::string locateFile(const std::string& filepathSuffix, const std::vector<std::string>& directories)
{
    const int MAX_DEPTH{10};
    bool found{false};
    std::string filepath;

    for (auto& dir : directories)
    {
        if (dir.back() != '/')
            filepath = dir + "/" + filepathSuffix;
        else
            filepath = dir + filepathSuffix;

        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(filepath);
            found = checkFile.is_open();
            if (found)
                break;
            filepath = "../" + filepath; // Try again in parent dir
        }

        if (found)
        {
            break;
        }

        filepath.clear();
    }

    if (filepath.empty())
    {
        std::string directoryList = std::accumulate(directories.begin() + 1, directories.end(), directories.front(),
                                                    [](const std::string& a, const std::string& b) { return a + "\n\t" + b; });
        std::cout << "Could not find " << filepathSuffix << " in data directories:\n\t" << directoryList << std::endl;
        std::cout << "&&&& FAILED" << std::endl;
        exit(EXIT_FAILURE);
    }
    return filepath;
}


//!
//! \brief  apply pre-processing actions on the image data (only standard deviation considered as the rest is done by the data conditioner)
//!
void applyResNet50StdevToDeviceImage(float32_t *deviceCHWImageData, const uint32_t width, const uint32_t height, cudaStream_t stream = 0);

#endif /* UTILSRESNET_H_ */
