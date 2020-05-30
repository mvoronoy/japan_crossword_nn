SET mypath=%~dp0
echo %mypath%

pushd %mypath%
cd ..

call ./venv/Scripts/activate.bat
start jupyter notebook
popd
:exit
