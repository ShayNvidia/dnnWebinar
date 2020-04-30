/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */


#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <framework/DriveWorksSample.hpp>
#include "utils.h"

// ------- driveworks includes ------------

// Context, SAL
#include <dw/core/Context.h>
#include <dw/core/VersionCurrent.h>


// IMAGE
#include <dw/interop/streamer/ImageStreamer.h>
#include <dw/imageprocessing/common/ImageProcessingCommon.h>
#include <dw/imageprocessing/geometry/imageTransformation/ImageTransformation.h>
#include <dw/imageprocessing/filtering/ImageFilter.h>
#include <dw/imageprocessing/filtering/Threshold.h>
// Renderer
#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Renderer.h>
#include <framework/WindowGLFW.hpp>
#include <framework/ProfilerCUDA.hpp>

// DNN
#include <dw/dnn/DNN.h>

#include "imageSource.h"

using namespace std;
/**
 * Class that holds functions and a variables common to all stereo samples
 */
using namespace dw_samples::common;


//!
//! \brief The sampleOnnxResNet class implements inference on classification networks.
//!
//! \details
//!
class sampleIntegrateDnn : public DriveWorksSample
{
public:
    const uint32_t WINDOW_HEIGHT = 600;
    const uint32_t WINDOW_WIDTH = 800;
    const float32_t C_MIN_CALSSIFICATION_ASSURANCE = 0.6f;
    const float32_t C_ROI_RECTANGLE_LINE_WITDH = 3.0f;
    const int32_t C_ROI_TEXT_HEIGHT = 30;
    const int32_t C_CLASSIFICATION_TEXT_X_OFFSET = 10;

    //!
	//! \brief the only constructor
	//!
    sampleIntegrateDnn(const ProgramArguments& args);

    // Sample framework override methods
    bool onInitialize() override;
    void onRender() override;
    void onResizeWindow(int width, int height) override;
    void onProcess() override;
    void onRelease() override;
    void onMouseDown(int button, float x, float y, int /* mods*/) override;
    void onMouseMove(float x, float y) override;
    void onMouseUp(int button, float /* x*/, float /* y*/, int /* mods*/) override;

    virtual void onKeyDown(int key, int scancode, int mods) override;
private:
    //!
	//! \brief initialize DW resources
	//!
    void initializeDriveWorks(dwContextHandle_t& context) const;
    //!
	//! \brief initialize all required resources running inference
	//!
    void initializeDNN();


    // ------------------------------------------------
    // Driveworks Context and SAL
    // ------------------------------------------------
    dwContextHandle_t               m_context           = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t  m_viz               = DW_NULL_HANDLE;
    dwRenderEngineHandle_t          m_renderEngine      = DW_NULL_HANDLE;
    dwRendererHandle_t              m_renderer          = DW_NULL_HANDLE;

    // ------------------------------------------------
    // DNN class members
    // ------------------------------------------------
    dwDNNHandle_t					m_dnn				= DW_NULL_HANDLE;
	dwDataConditionerHandle_t 		m_dataConditioner 	= DW_NULL_HANDLE;
	cublasHandle_t 				m_cublasHandle		= nullptr;
    cudaStream_t 					m_cudaStream 		= 0;
    uint32_t 						m_totalSizeInput	= 0;
	uint32_t 						m_totalSizeOutput 	= 0;
	dwRect 						m_detectionRoi;

    // DNN variables
	string m_trtModelFile;
	std::string m_imageSourceName;
	int m_currentClassificationIndex = -1;
	float m_currentClassificationProb = 0.0f;
    string m_referenceFileName;
	std::vector<std::string> m_referenceVector;
	dwBlobSize m_networkInputDimensions;
	dwBlobSize m_networkOutputDimensions;

	// image variables
	imageSource m_imageSource;
	// streamer from CUDA to GL, used for display
	dwImageStreamerHandle_t m_streamerCUDA2GL = DW_NULL_HANDLE;
	// CUDA image with RGBA format that we will use
	dwImageHandle_t m_rgbaImageHandleCUDA = DW_NULL_HANDLE;

	 // class cuda buffers
	std::unique_ptr<ManagedCudaBuffer<float>> 			m_cudaBuffersIn;
	std::unique_ptr<ManagedCudaBuffer<float>> 			m_cudaBuffersOut;

	// ROI variables
	bool m_updateBoxNeeded = false;
	dwBox2Df m_roiSelection = {0,0,0,0};
	dwRect boxToRec(dwBox2Df& b) {return {static_cast<int>(b.x), static_cast<int>(b.y), static_cast<int>(b.width), static_cast<int>(b.height)};}
};

//#######################################################################################
sampleIntegrateDnn::sampleIntegrateDnn(const ProgramArguments& args)  : DriveWorksSample(args), m_imageSource(getArgs())
{
	vector<string> dataDirs;
	dataDirs.push_back(getArgument("data"));
	dataDirs.push_back("data/samples/resnet50/");
	dataDirs.push_back("data/resnet50/");
	dataDirs.push_back(getArgument("imagesDir"));
	dataDirs.push_back("./"); //! In case of absolute path search
	log ("Validating input parameters. Using following input files for inference.\n" );
	m_trtModelFile = locateFile(getArgument("tensorRT_model"), dataDirs);
	log ("    TensorRT model File: %s\n" ,  m_trtModelFile.c_str());
	m_referenceFileName = locateFile(getArgument("referenceFileName"), dataDirs);
	log("    Reference File: %s\n" , m_referenceFileName.c_str() );
}

//#######################################################################################
void sampleIntegrateDnn::onKeyDown(int key, int scancode, int mods)
{
		(void)scancode;
		(void)mods;
		(void)key;
		if (GLFW_KEY_L == key || 'l' == key)
		{
			m_imageSourceName = m_imageSource.loadNewImageFile();
			m_currentClassificationIndex = -1;
			// Detection region
			m_detectionRoi.width = m_roiSelection.width = m_imageSource.currentImageWidth();
			m_detectionRoi.height = m_roiSelection.height = m_imageSource.currentImageHeight();
			m_detectionRoi.x = m_roiSelection.x = 0;
			m_detectionRoi.y = m_roiSelection.y = 0;
		}
		if ('q' == key)
		{
			stop();
		}
}

//#######################################################################################
void sampleIntegrateDnn::onMouseDown(int button, float x, float y, int /* mods*/)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT)
	{
		m_updateBoxNeeded = true;
		m_roiSelection.x = x * m_imageSource.currentImageWidth() / getWindowWidth();
		m_roiSelection.y = y * m_imageSource.currentImageHeight() / getWindowHeight();
		m_roiSelection.width = 0;
		m_roiSelection.height = 0;
	}
}

//#######################################################################################
void sampleIntegrateDnn::onMouseMove(float x, float y)
{
	if (!m_updateBoxNeeded)
		return;

	int32_t fx    = x * m_imageSource.currentImageWidth() / getWindowWidth();
	int32_t fy    = y * m_imageSource.currentImageHeight() / getWindowHeight();

	m_roiSelection.width     = abs(fx - m_roiSelection.x);
	m_roiSelection.height    = abs(fy - m_roiSelection.y);
	if (m_roiSelection.x > fx)
		m_roiSelection.x = fx;
	if (m_roiSelection.y > fy)
		m_roiSelection.y = fy;
}

//#######################################################################################
void sampleIntegrateDnn::onMouseUp(int button, float /* x*/, float /* y*/, int /* mods*/)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT)
	{
		if (m_updateBoxNeeded)
		{
			m_updateBoxNeeded = false;

			if (m_roiSelection.width > 1.0f && m_roiSelection.height > 1.0f
					&& (m_roiSelection.width + m_roiSelection.x) <= m_imageSource.currentImageWidth()
					&& (m_roiSelection.height + m_roiSelection.y) <= m_imageSource.currentImageHeight()
					&& m_roiSelection.x >= 0.0f
					&& m_roiSelection.y >= 0.0f)
			{

				m_detectionRoi = boxToRec(m_roiSelection);
			}
			else
			{
				m_detectionRoi.x = m_roiSelection.x = m_detectionRoi.y = m_roiSelection.y = 0;
				m_detectionRoi.width = m_roiSelection.width = m_imageSource.currentImageWidth();
				m_detectionRoi.height = m_roiSelection.height = m_imageSource.currentImageHeight();
			}

		}
	}
}

//#######################################################################################
void sampleIntegrateDnn::initializeDriveWorks(dwContextHandle_t& context) const
{
	// initialize logger to print verbose message on console in color
	CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
	CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));

	// initialize SDK context, using data folder
	dwContextParameters sdkParams = {};
	sdkParams.dataPath            = DataPath::get_cstr();

	#ifdef VIBRANTE
	sdkParams.eglDisplay = getEGLDisplay();
	#endif

	CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, &sdkParams));
}

//#######################################################################################
void sampleIntegrateDnn::initializeDNN()
{

	// Initialize cublas handle to run max value on output (post-processing)
	cudaStreamCreate(&m_cudaStream);
	CHECK_CUBLAS_ERROR(cublasCreate(&m_cublasHandle));
	CHECK_CUBLAS_ERROR(cublasSetStream(m_cublasHandle, m_cudaStream));

	// Initialize DNN from a TensorRT file
	CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFileNew(&m_dnn, m_trtModelFile.c_str(), nullptr,DW_PROCESSOR_TYPE_DLA_0, m_context));
	CHECK_DW_ERROR(dwDNN_setCUDAStream(m_cudaStream, m_dnn));
	// Get input and output dimensions
	CHECK_DW_ERROR(dwDNN_getInputSize(&m_networkInputDimensions, 0U, m_dnn));
	CHECK_DW_ERROR(dwDNN_getOutputSize(&m_networkOutputDimensions, 0U, m_dnn));

	auto getTotalSize = [](const dwBlobSize& blobSize) {
		return blobSize.channels * blobSize.height * blobSize.width;
	};

	// Calculate total size needed to store input and output
	m_totalSizeInput      = getTotalSize(m_networkInputDimensions);
	m_totalSizeOutput = getTotalSize(m_networkOutputDimensions);

	m_cudaBuffersIn.reset(new ManagedCudaBuffer<float>(m_totalSizeInput));
	m_cudaBuffersOut.reset(new ManagedCudaBuffer<float>(m_totalSizeOutput));
	// Get metadata from DNN module
	// DNN loads metadata automatically from json file stored next to the dnn model,
	// with the same name but additional .json extension if present.
	// Otherwise, the metadata will be filled with default values and the dataconditioner parameters
	// should be filled manually.
	dwDNNMetaData metadata;
	CHECK_DW_ERROR(dwDNN_getMetaData(&metadata, m_dnn));

	// Initialie data conditioner
	CHECK_DW_ERROR(dwDataConditioner_initialize(	&m_dataConditioner,
											&m_networkInputDimensions,
											&metadata.dataConditionerParams, m_cudaStream,
											m_context));

}


//#######################################################################################
bool sampleIntegrateDnn::onInitialize()
{
	// -----------------------------------------
	// Initialize DriveWorks context and SAL
	// -----------------------------------------
	{
		initializeDriveWorks(m_context);
		dwVisualizationInitialize(&m_viz, m_context);
	}


	// -----------------------------------------
	// Initialize DNN
	// -----------------------------------------
	{
		initializeDNN();
		// -----------------------------------
		//	Read the input data
		// -----------------------------------
		if (!readReferenceFile(m_referenceFileName, m_referenceVector))
		{
			logError ( "Unable to read reference file: " , getArgument("referenceFileName") );
			return false;
		}
	}
    // -----------------------------
    // Initialize Renderer
    // -----------------------------
    {
        // init render engine with default params
        dwRenderEngineParams params{};
        CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params, getWindowWidth(), getWindowHeight()));
        CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_viz));

        dwRenderer_initialize(&m_renderer, m_viz);
        dwRect rect;
        rect.width  = getWindowWidth();
        rect.height = getWindowHeight();
        rect.x      = 0;
        rect.y      = 0;
        dwRenderer_setRect(rect, m_renderer);
    }

    // Initialize image source handler
    // -----------------------------
    m_imageSource.initialize(m_context);

    // Initialize image and GL data
    // -----------------------------
    {
		// create a new imageStreamer to stream CUDA to GL and get a openGL texture to render on screen
		dwImageProperties cudaProp;
		cudaProp.width = m_imageSource.maxImageWidth();
		cudaProp.height = m_imageSource.maxImageHeight();
		cudaProp.format = DW_IMAGE_FORMAT_RGBA_UINT8;
		cudaProp.type = DW_IMAGE_CUDA;
		cudaProp.memoryLayout = DW_IMAGE_MEMORY_TYPE_DEFAULT; // not necessary to specify if init with {}
		// instantiation of an image streamer that can pass images to OpenGL.
		CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_streamerCUDA2GL, &cudaProp, DW_IMAGE_GL, m_context));
    }

    m_imageSourceName = m_imageSource.loadNewImageFile();
	m_currentClassificationIndex = -1;
	// Detection region
	m_detectionRoi.width = m_roiSelection.width = m_imageSource.currentImageWidth();
	m_detectionRoi.height = m_roiSelection.height = m_imageSource.currentImageHeight();
	m_detectionRoi.x = m_roiSelection.x = 0;
	m_detectionRoi.y = m_roiSelection.y = 0;
    return true;
}


//#######################################################################################
void sampleIntegrateDnn::onResizeWindow(int width, int height)
{
	CHECK_DW_ERROR(dwRenderEngine_reset(m_renderEngine));
	//dwRectf rect;
	dwRect rect;
	rect.width  = width;
	rect.height = height;
	rect.x      = 0;
	rect.y      = 0;
	dwRenderer_setRect(rect, m_renderer);
}



//#######################################################################################
void sampleIntegrateDnn::onProcess()
{
	// Run data conditioner to prepare input for the network
	if(m_imageSource.getCudaImageRGBA(m_rgbaImageHandleCUDA, getProfilerCUDA()))
	{
		dwImageCUDA* rgbaImage;
		CHECK_DW_ERROR(dwImage_getCUDA(&rgbaImage, m_rgbaImageHandleCUDA));
		float32_t * deviceInputData = m_cudaBuffersIn->getDeviceBuffer();
		float32_t * deviceOutputData = m_cudaBuffersOut->getDeviceBuffer();
		// Run data conditioner on the input to apply pre-processing actions on the input image
		{
			dw::common::ProfileCUDASection s(getProfilerCUDA(), "preprocessingFrame");
			CHECK_DW_ERROR(dwDataConditioner_prepareData(deviceInputData,  &rgbaImage, 1, &m_detectionRoi, cudaAddressModeClamp, m_dataConditioner));
			// NOTICE: resnet50 requires applying StdDev to the image channels as a pre-processing phase
			//			dataConditioner currently does not support it, will be added in the future release
			applyResNet50StdevToDeviceImage(deviceInputData, m_networkInputDimensions.width, m_networkInputDimensions.height, m_cudaStream);
			// wait for pre-processing to finish (when not profiling is not necessary)
			cudaStreamSynchronize(m_cudaStream);
		}

		// Run DNN inference on the output of data conditioner
		{
			dw::common::ProfileCUDASection s(getProfilerCUDA(), "inferenceFrame");
			CHECK_DW_ERROR(dwDNN_infer(&deviceOutputData, &deviceInputData , 1U, m_dnn));
			// wait for inference to finish
			cudaStreamSynchronize(m_cudaStream);
		}

		// calculate post-processing result
		int maxIdx = 0;
		CHECK_CUBLAS_ERROR(cublasIsamax(m_cublasHandle, m_totalSizeOutput, deviceOutputData, 1, &maxIdx));
		if(0 != maxIdx)
		{
			m_currentClassificationIndex = maxIdx - 1;
			cudaMemcpy(&m_currentClassificationProb, deviceOutputData + m_currentClassificationIndex, sizeof(float), cudaMemcpyDeviceToHost);
		}
	}
}



//#######################################################################################
void sampleIntegrateDnn::onRender()
{
	CHECK_DW_ERROR(dwRenderer_setColor({1.f, 1.f, 1.f, 1.f}, m_renderer));
	CHECK_DW_ERROR(dwImageStreamerGL_producerSend(m_rgbaImageHandleCUDA, m_streamerCUDA2GL));
	{
		dwImageHandle_t glImageHandler;
		// receive a dwImageGL that we can render
		CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&glImageHandler, 3000, m_streamerCUDA2GL));
		// render
		{
			dwImageGL *glImage;
			CHECK_DW_ERROR(dwImage_getGL(&glImage, glImageHandler));

			glClearColor(0.0, 0.0, 0.0, 0.0);
			glClear(GL_COLOR_BUFFER_BIT);
			// calculate what part of the max size image contains the image to display
			float32_t ratioWidth = static_cast<float32_t>(m_imageSource.currentImageWidth()) / static_cast<float32_t>(m_imageSource.maxImageWidth());
			float32_t ratioHeight = static_cast<float32_t>(m_imageSource.currentImageHeight()) / static_cast<float32_t>(m_imageSource.maxImageHeight());
			// convert values to range of (-1,1)
			float32_t maxX = (ratioWidth * 2.0f) - 1.0f;
			float32_t maxY = (ratioHeight * 2.0) - 1.0f;

			// leaving space for labels display
			dwRect rect;
			rect.x = 0;
			rect.y = 50;
			rect.width = getWindowWidth();
			rect.height = getWindowHeight() - rect.y - C_ROI_TEXT_HEIGHT;
			CHECK_DW_ERROR(dwRenderer_setRect(rect,m_renderer));
			CHECK_DW_ERROR(dwRenderer_renderSubTexture(glImage->tex, glImage->target, -1.0f, -1.0f, maxX, maxY,m_renderer));
			// setting back drawing area to full screen
			rect.y = 0;
			rect.height = getWindowHeight();
			CHECK_DW_ERROR(dwRenderer_setRect(rect,m_renderer));

		}
		// return the received gl since we don't use it anymore
		CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&glImageHandler, m_streamerCUDA2GL));
	}
	// wait to get back the cuda image we posted in the cuda->gl stream. We will receive a pointer to it and,
	// to be sure we are getting back the same image we posted, we compare the pointer to our dwImageCPU
	CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 1000, m_streamerCUDA2GL));

	// render the classification result labels
	CHECK_DW_ERROR(dwRenderer_setFont(DW_RENDER_FONT_VERDANA_24, m_renderer));
	CHECK_DW_ERROR(dwRenderer_renderText(C_CLASSIFICATION_TEXT_X_OFFSET, 20, m_imageSourceName.c_str(), m_renderer));
	if(m_currentClassificationIndex != -1)
	{
		static int prevClassification = -1;
		std::string classificationMessage = "No Classification";
		//if(m_currentClassificationProb > 0.4f)
		{
			classificationMessage =
				std::string("classification result is: '") +
				m_referenceVector[m_currentClassificationIndex] +
				"' ,prob =" + std::to_string(m_currentClassificationProb);
		}
		CHECK_DW_ERROR(dwRenderer_renderText(C_CLASSIFICATION_TEXT_X_OFFSET, 0, classificationMessage.c_str(), m_renderer));
		if(prevClassification != m_currentClassificationIndex)
		{
			prevClassification = m_currentClassificationIndex;
			log(classificationMessage.c_str());
			log("\n");
		}
	}
	// render ROI
	{
		std::string roiString = std::to_string(static_cast<int>(m_roiSelection.width)) + "x" + std::to_string(static_cast<int>(m_roiSelection.height)) + " Click and drag to select ROI";
		CHECK_DW_ERROR(dwRenderer_renderText(10 , getWindowHeight() - C_ROI_TEXT_HEIGHT ,roiString.c_str() , m_renderer));
		CHECK_DW_ERROR(dwRenderer_setLineWidth(C_ROI_RECTANGLE_LINE_WITDH, m_renderer));
		std::vector<dwVector2f> roiPoints;
		roiPoints.push_back({m_roiSelection.x / m_imageSource.currentImageWidth()							, m_roiSelection.y / m_imageSource.currentImageHeight()});
		roiPoints.push_back({(m_roiSelection.x + m_roiSelection.width) / m_imageSource.currentImageWidth()	, m_roiSelection.y / m_imageSource.currentImageHeight()});
		roiPoints.push_back({(m_roiSelection.x + m_roiSelection.width) / m_imageSource.currentImageWidth()	, (m_roiSelection.y + m_roiSelection.height) / m_imageSource.currentImageHeight()});
		roiPoints.push_back({m_roiSelection.x / m_imageSource.currentImageWidth()							, (m_roiSelection.y + m_roiSelection.height) / m_imageSource.currentImageHeight()});
		CHECK_DW_ERROR(dwRenderer_renderData2D( roiPoints.data(), roiPoints.size(), DW_RENDER_PRIM_LINELOOP, m_renderer));
	}
}

//#######################################################################################
void sampleIntegrateDnn::onRelease()
{
	if(DW_NULL_HANDLE != m_dnn)
	{
		CHECK_DW_ERROR(dwDNN_release(m_dnn));
	}

	if(DW_NULL_HANDLE != m_dataConditioner)
	{
		CHECK_DW_ERROR(dwDataConditioner_release(m_dataConditioner));
	}

	if( 0 != m_cudaStream)
	{
		CHECK_CUDA_ERROR(cudaStreamDestroy(m_cudaStream));
	}

	if (DW_NULL_HANDLE != m_renderEngine)
	{
		CHECK_DW_ERROR(dwRenderEngine_release(m_renderEngine));
	}

	dwRenderer_release(m_renderer);

	m_imageSource.onRelease();

	// release streamer to GL
	if(DW_NULL_HANDLE != m_streamerCUDA2GL)
	{
		CHECK_DW_ERROR(dwImageStreamerGL_release(m_streamerCUDA2GL));
		m_streamerCUDA2GL = DW_NULL_HANDLE;
	}

	// -----------------------------------
	// Release SDK
	// -----------------------------------
	CHECK_DW_ERROR(dwVisualizationRelease(m_viz));
	CHECK_DW_ERROR(dwRelease(m_context));
	CHECK_DW_ERROR(dwLogger_release());

}

//#########################################
//
//		------------ MAIN ---------------
//
//#########################################
int main(int argc, const char **argv)
{
	// define all arguments used by the application
	ProgramArguments args(argc, argv,
	{
			ProgramArguments::Option_t("tensorRT_model", nullptr, (std::string("path to TensorRT model file.").c_str())),
			ProgramArguments::Option_t("referenceFileName", "reference_labels.txt", (std::string("reference file for classification labels.").c_str())),
			ProgramArguments::Option_t("data", "data/samples/resnet50/",(std::string("Specify data directory to search for above files in case absolute paths to files are not provided.").c_str())),
			ProgramArguments::Option_t("imagesDir", "", (std::string("path to ppm images folder.").c_str())),
			ProgramArguments::Option_t("imageFile", "", (std::string("ppm image file to proccess.").c_str())),
			ProgramArguments::Option_t("useCamera", "0", (std::string("specify whether to use a USB camera as source.").c_str())),
#ifdef VIBRANTE
			ProgramArguments::Option_t("camera-type", "ar0231-rccb-bae-sf3324", "camera gmsl type (see sample_camera_gmsl for more info), usb"),
			ProgramArguments::Option_t("camera-group", "a", "input port"),
			ProgramArguments::Option_t("camera-index", "0", "camera index within the camera-group 0-3"),
#else
			ProgramArguments::Option_t("camera-type", "usb", "only usb type is available on host"),
#endif
			ProgramArguments::Option_t("usbDeviceID", "0", (std::string("the device ID of the camera.")).c_str()),
			ProgramArguments::Option_t("usbCamMode", "0", (std::string("Applicable for generic camera only. Specifies a method for selecting capture settings: \n"
																"`a`: choose mode with maximum resolution \n"
																"`b`: choose mode with maximum fps \n"
																"integer number: choose mode by index\n")).c_str())
	},
	"illustrating how to use a dnn network with DriveWroks");

    sampleIntegrateDnn app(args);
    if(args.enabled("offscreen"))
    {
    	app.initializeCommandLineInput();
    }
    app.initializeWindow("import dnn Sample", app.WINDOW_WIDTH, app.WINDOW_HEIGHT, args.enabled("offscreen"));
    return app.run();
}
