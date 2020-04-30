# dnnWebinar
NVIDIA DriveWorks sample code for webinar "Integrating DNN Inference into Autonomous Vehicle Applications with NVIDIA DriveWorks SDK"

==========
Content:
==========
1. data - folder containing all the data needed for this sample, and some images examples
2. sources\src\dnn\ - folder containing the sample code to be compiled for execution (on the host and on the Drive AGX taget)

==========
How to:
==========
1. create an optimized DW TRT plan file (on the Drive AGX Drive target and on the host):

	a. cd data/resnet50

	b. /usr/local/driveworks/tools/dnn/tensorRT_optimization --modelType=onnx --onnxFile=resnet50.onnx --out=resnet50Onnx2TRTx86.bin


2. compile the sample code:

	a. copy the DW samples into a folder

	b. than merge and replace the samples folder with this reposotiroty samples folder.

	c. follow compilation instructions in tutorial 1 of DW documentation for compilation for the host and for the Drive AGX


3. execute the sample. for example:

./sample_integrate_dnn --tensorRT_model=/path/to/generated/plan/file/resnet50Onnx2TRTx86.bin --imageFile=/path/to/the/images/in/data/folder/image2.ppm --data=/path/to/the/folder/data


good luck!

For any questions, please post it on the forum.
