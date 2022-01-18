"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\msbuild" build/cuda_opengl_interop.sln
ROBOCOPY build/Debug/ .. cuda_opengl_interop.pyd
python test.py