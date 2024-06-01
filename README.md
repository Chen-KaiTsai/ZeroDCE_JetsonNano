# ZeroDCE_JetsonNano

A customed ZeroDCE implementation on Jetson Nano with CSI camera. 

I build this small project on a Jetson Nano 2G with a IMX219.

# Contain

* A Makefile is provided.
* A weight file is provided.
* A `.pt` file provided. This include the entire model.
* If you want to export a weight file, please refer to the `export_weight.py` and `qWeight.h` file.

# Result Example
## Original Input from CSI Camera
![image](https://github.com/Chen-KaiTsai/ZeroDCE_JetsonNano/blob/main/imgs/testInput.png)
## Enhanced Output
![image](https://github.com/Chen-KaiTsai/ZeroDCE_JetsonNano/blob/main/imgs/Enhanced_CPP_output.png)
