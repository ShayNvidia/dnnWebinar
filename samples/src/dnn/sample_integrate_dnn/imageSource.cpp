#include <dirent.h>
#include "imageSource.h"
#include "utils.h"
#include <framework/Log.hpp>
#include <framework/Checks.hpp>


//#######################################################################################
imageSource::imageSource(ProgramArguments& args)
{
	m_useCameraSource = args.get("useCamera").compare("1") == 0;
	m_useUsbCamera = args.get("camera-type").compare("usb") == 0;

	loadListOfImageFiles(args);
	if(m_useCameraSource)
	{
		if(m_useUsbCamera)
		{
			m_sensorParameters = "device=" + args.get("usbDeviceID");
			const std::string& modeParam = args.get("usbCamMode");
			if(!modeParam.empty())
			{
				m_sensorParameters += ",mode=" + modeParam;
			}
			m_sensorProtocol = "camera.usb";
		}
		else
		{
			m_sensorParameters = std::string("output-format=yuv");
			m_sensorParameters += std::string(",camera-type=") + args.get("camera-type");
			m_sensorParameters += std::string(",camera-group=") + args.get("camera-group");
			std::string cameraMask[4] = {"0001", "0010", "0100", "1000"};
			uint32_t cameraIdx = std::stoi(args.get("camera-index"));
			if (cameraIdx < 0 || cameraIdx > 3)
			{
				logError("Error: camera index must be 0, 1, 2 or 3\n");
				throw std::runtime_error("invalid camera index!");
			}
			m_sensorParameters += std::string(",camera-mask=") + cameraMask[cameraIdx];
			m_sensorProtocol             = "camera.gmsl";
		}
	}
}

//#######################################################################################
void imageSource::initialize(dwContextHandle_t context)
{
	m_context = context;
	if(m_useCameraSource)
	{
		// ----------------------------------------
		// Initialize camera and streaming
		// ----------------------------------------
		initializeCameraProccessing(context);
	}
	else
	{
		// ----------------------------------------
		// Initialize first image loading
		// ----------------------------------------
		if ( 0 == m_imagefilesToLoad.size())
		{
			throw std::runtime_error("no image files selected, select either an image to load or a camera as source");
		}
		dwImageProperties propsGpu;
		propsGpu.width = m_maxImageWidth;
		propsGpu.height = m_maxImageHeight;
		propsGpu.format = DW_IMAGE_FORMAT_RGBA_UINT8;
		propsGpu.memoryLayout = DW_IMAGE_MEMORY_TYPE_DEFAULT; // not necessary to specify if init with {}
		propsGpu.type = DW_IMAGE_CUDA;
		CHECK_DW_ERROR(dwImage_create(&m_imageGpuRGBA, propsGpu, m_context));
		propsGpu.format = DW_IMAGE_FORMAT_RGB_UINT8;
		CHECK_DW_ERROR(dwImage_create(&m_imageFileRGBHandleGPU, propsGpu, m_context));
	}

}


//#######################################################################################
std::string imageSource::loadNewImageFile()
{
	if(!m_useCameraSource)
	{
		log("reading image file: %s\n", m_imagefilesToLoad[m_nextImageFileToLoad].c_str());
		PPM ppm;
		if(!readPPMFile(m_imagefilesToLoad[m_nextImageFileToLoad], ppm))
		{
			log("reading image file: %s\n", m_imagefilesToLoad[m_nextImageFileToLoad].c_str());
			logError("File cannot be opened or format error! %s", m_imagefilesToLoad[m_nextImageFileToLoad].c_str());
			m_nextImageFileToLoad = (m_nextImageFileToLoad + 1) % m_imagefilesToLoad.size();
		}
		else
		{
			if (ppm.w > m_maxImageWidth || ppm.h > m_maxImageHeight)
			{
				throw std::runtime_error("image is too big than maximum size");
			}
			unsigned char* imageColorsRGB = ppm.buffer.data();
			m_currentImageWidth = ppm.w;
			m_currentImageHeight = ppm.h;
			m_currentImageFile = ppm.fileName;
			m_nextImageFileToLoad = (m_nextImageFileToLoad + 1) % m_imagefilesToLoad.size();
			dwImageCUDA* pImageDataRGBGPU = nullptr;
			dwImage_getCUDA(&pImageDataRGBGPU, m_imageFileRGBHandleGPU);
			for(int ch = 0 ; ch < DW_MAX_IMAGE_PLANES ; ch++)
			{
				if(pImageDataRGBGPU->pitch[ch] > 0)
				{
					unsigned int targetPitch = pImageDataRGBGPU->pitch[ch];
					unsigned int targetRow = 0;
					for (; targetRow < m_currentImageHeight ; targetRow++)
					{
						cudaMemcpy(&(reinterpret_cast<char*>(pImageDataRGBGPU->dptr[ch])[targetRow*targetPitch]),&imageColorsRGB[ targetRow * m_currentImageWidth * 3], m_currentImageWidth * 3, cudaMemcpyHostToDevice);
					}
					for (unsigned int remaintargetRow = targetRow ; remaintargetRow < m_currentImageHeight ; remaintargetRow++)
					{
						cudaMemset(&(reinterpret_cast<char*>(pImageDataRGBGPU->dptr[ch])[remaintargetRow*targetPitch]), 0, targetPitch);
					}
				}
			}
			//format convert the RGB into RGBA
			CHECK_DW_ERROR(dwImage_copyConvert(m_imageGpuRGBA, m_imageFileRGBHandleGPU, m_context));
		}
	}
	std::string source;
	if(m_useCameraSource)
	{
		source = "camera source:";
	}
	else
	{
		source = "image: " + m_currentImageFile;
	}
	return source;
}



//#######################################################################################
void imageSource::initializeCameraProccessing(dwContextHandle_t context)
{
	CHECK_DW_ERROR(dwSAL_initialize(&m_sal, context));
	// initialize the sensor
	{
		dwSensorParams sensorParams;
		sensorParams.protocol = m_sensorProtocol.c_str();
		sensorParams.parameters = m_sensorParameters.c_str();
		CHECK_DW_ERROR(dwSAL_createSensor(&m_camera, sensorParams, m_sal));
	}

	// Log available modes capture modes
	{
		uint32_t numModes = 0;
		CHECK_DW_ERROR(dwSensorCamera_getNumSupportedCaptureModes(&numModes, m_camera));

		if(numModes > 1)
		{
			dwCameraProperties properties{};
			CHECK_DW_ERROR(dwSensorCamera_getSensorProperties(&properties, m_camera));

			for (uint32_t modeIdx = 0; modeIdx < numModes; ++modeIdx)
			{
				dwCameraProperties mode{};
				CHECK_DW_ERROR(dwSensorCamera_getSupportedCaptureMode(&mode, modeIdx, m_camera));

				const char* msgEnd = (mode.framerate == properties.framerate &&
									  mode.resolution.x == properties.resolution.x &&
									  mode.resolution.y == properties.resolution.y) ? " fps (*)" : " fps";

				std::cout << "Mode " << modeIdx << ": " << mode.resolution.x << "x" << mode.resolution.y << " " << mode.framerate << msgEnd << std::endl;
			}
		}
	}

	// Retrieve camera dimensions
	CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&m_cameraImageProps, DW_CAMERA_OUTPUT_NATIVE_PROCESSED, m_camera));

	dwImageProperties cudaProp = m_cameraImageProps;
	cudaProp.type = DW_IMAGE_CUDA;
	cudaProp.format = DW_IMAGE_FORMAT_RGBA_UINT8;
	// create the image handler to hold the image comming from the camera
	CHECK_DW_ERROR(dwImage_create(&m_imageGpuRGBA, cudaProp, context));

	std::cout << "Camera image with " << m_cameraImageProps.width << "x" << m_cameraImageProps.height << std::endl;
	m_maxImageWidth = m_currentImageWidth = m_cameraImageProps.width;
	m_maxImageHeight  = m_currentImageHeight = m_cameraImageProps.height;

	// start sensor
	CHECK_DW_ERROR(dwSensor_start(m_camera));

}


//#######################################################################################
void imageSource::loadListOfImageFiles(ProgramArguments& args)
{
	std::string imagesPath = args.get("imagesDir");
	m_imageFile = args.get("imageFile");
	if(m_imageFile != "")
	{
		m_imagefilesToLoad.push_back(m_imageFile);
		log("Found %d images.\n", m_imagefilesToLoad.size());
	}
	else if(imagesPath != "")
	{
		DIR *dir;
		struct dirent * entry;
		if ((dir = opendir (imagesPath.c_str())) != NULL)
		{
			/* print all the files and directories within directory */
			while ((entry = readdir (dir)) != NULL)
			{
				std::string imageToOpen = entry->d_name;
				if(std::string::npos != imageToOpen.find(".ppm"))
				{
					m_imagefilesToLoad.push_back(imagesPath+imageToOpen);
				}
			}
			closedir (dir);
		}
		log("Found %d images.\n", m_imagefilesToLoad.size());
	}
}

//#######################################################################################
dwImageHandle_t imageSource::getNewCameraImage()
{
	dwImageHandle_t cameraImage               = DW_NULL_HANDLE;
	const dwStatus result = dwSensorCamera_readFrame(&m_frame, 0, 100000, m_camera);
	if (DW_TIME_OUT == result) {
		return DW_NULL_HANDLE;
	}else if (DW_NOT_AVAILABLE == result) {
		std::cerr << "Camera is not running or not found" << std::endl;
		return DW_NULL_HANDLE;
	}else if (DW_SUCCESS != result) {
		std::cerr << "Cannot get frame from the camera: " << dwGetStatusName(result) << std::endl;
		return DW_NULL_HANDLE;
	}
	dwCameraOutputType outputType = DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8;
	CHECK_DW_ERROR(dwSensorCamera_getImage(&cameraImage, outputType, m_frame));
	return cameraImage;
}

//#######################################################################################
bool imageSource::getCudaImageRGBA(dwImageHandle_t& rgbaImageHandleCUDA, dw::common::ProfilerCUDA* profiler)
{
	bool newImageReady = false;
	if(m_useCameraSource)
	{
		dw::common::ProfileCUDASection s(profiler, "cameraImageConsuming");
		dwImageHandle_t imageHandler = getNewCameraImage();
		if(DW_NULL_HANDLE != imageHandler)
		{
			CHECK_DW_ERROR(dwImage_copyConvert(m_imageGpuRGBA, imageHandler, m_context));
			rgbaImageHandleCUDA = m_imageGpuRGBA;
			// return frame from camera (if receive)
			if(m_frame)
			{
				CHECK_DW_ERROR(dwSensorCamera_returnFrame(&m_frame));
			}
			newImageReady = true;
		}
	}
	else
	{
		newImageReady = true;
		rgbaImageHandleCUDA = m_imageGpuRGBA;
	}
	return newImageReady;
}

//#######################################################################################
void imageSource::onRelease()
{
	if(m_useCameraSource)
	{
		if(DW_NULL_HANDLE != m_camera)
		{
			CHECK_DW_ERROR(dwSensor_stop(m_camera));
			CHECK_DW_ERROR(dwSAL_releaseSensor(m_camera));
			m_camera = DW_NULL_HANDLE;
		}
		if(DW_NULL_HANDLE != m_sal)
		{
			CHECK_DW_ERROR(dwSAL_release(m_sal));
			m_sal = DW_NULL_HANDLE;
		}
	}
	else
	{
		if(DW_NULL_HANDLE != m_imageFileRGBHandleGPU)
		{
			CHECK_DW_ERROR(dwImage_destroy(m_imageFileRGBHandleGPU));
			m_imageFileRGBHandleGPU = DW_NULL_HANDLE;
		}
	}
	if (m_imageGpuRGBA)
	{
		CHECK_DW_ERROR(dwImage_destroy(m_imageGpuRGBA));
		m_imageGpuRGBA  = DW_NULL_HANDLE;
	}
}

imageSource::~imageSource()
{
	onRelease();
}
