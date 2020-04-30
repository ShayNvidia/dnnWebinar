#ifndef IMAGE_H_
#define IMAGE_H_


#include <framework/ProgramArguments.hpp>
#include <framework/ProfilerCUDA.hpp>

// Context, SAL
#include <dw/core/Context.h>
#include <dw/sensors/Sensors.h>
#include <dw/sensors/camera/Camera.h>

// IMAGE
#include <dw/interop/streamer/ImageStreamer.h>
#include <dw/imageprocessing/common/ImageProcessingCommon.h>
#include <dw/imageprocessing/geometry/imageTransformation/ImageTransformation.h>
#include <dw/imageprocessing/filtering/ImageFilter.h>
#include <dw/imageprocessing/filtering/Threshold.h>



class imageSource
{
private:
	dwContextHandle_t m_context              = DW_NULL_HANDLE;

	// Camera variables
	bool m_useCameraSource = false;
	bool m_useUsbCamera = false;
	const uint32_t MAX_CAPTURE_MODES            = 1024u;
	dwSALHandle_t m_sal                         = DW_NULL_HANDLE;
	dwSensorHandle_t m_camera                   = DW_NULL_HANDLE;
	dwCameraFrameHandle_t m_frame               = DW_NULL_HANDLE;
	dwImageProperties m_cameraImageProps;
	std::string m_sensorParameters;
	std::string m_sensorProtocol;


	// loading image variables
	std::string m_imageFile;
	std::string m_currentImageFile;
	std::vector<std::string> m_imagefilesToLoad;
	size_t m_nextImageFileToLoad = 0;

	unsigned int m_maxImageWidth = 1280;
	unsigned int m_maxImageHeight = 1280;

	unsigned int m_currentImageWidth = 0;
	unsigned int m_currentImageHeight = 0;

	// ------------------------------------------------
	// Image Streaming class members
	// ------------------------------------------------
	// GPU image with RGB format - that will be converted to RGBA
	dwImageHandle_t m_imageFileRGBHandleGPU = DW_NULL_HANDLE;
	// GPU image with RGBA format
	dwImageHandle_t m_imageGpuRGBA = DW_NULL_HANDLE;

    //!
	//! \brief initialize all required resources for cmaera images processing
	//!
    void initializeCameraProccessing(dwContextHandle_t context);
    //!
	//! \brief initialize all required resources for image files processing
	//!
	void initializeImageFileProcessing(dwContextHandle_t context);
	//!
	//! \brief loads all the images in the given folder (if given)
	//!
	void loadListOfImageFiles(ProgramArguments& args);
	//!
	//! \brief gets a new image form the camera
	//!
	dwImageHandle_t getNewCameraImage();
public:
	imageSource(ProgramArguments& args);
	~imageSource();
	void initialize(dwContextHandle_t context);
	std::string loadNewImageFile();
	bool getCudaImageRGBA(dwImageHandle_t& rgbaImageHandleCUDA,  dw::common::ProfilerCUDA* profiler = nullptr);
	void onRelease();
	unsigned int currentImageWidth() {return m_currentImageWidth;}
	unsigned int currentImageHeight() {return m_currentImageHeight;}
	unsigned int maxImageWidth() {return m_maxImageWidth;}
	unsigned int maxImageHeight() {return m_maxImageHeight;}
};



#endif /* IMAGE_H_ */
