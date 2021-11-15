# Topopt Comec

**GUI for topology optimization of compliant mechanisms coded in Python by Luc PREVOST.**

![default](https://user-images.githubusercontent.com/52052772/139711327-5e9393f3-7dc1-4785-b9c7-20748fd9e566.png)

📃 INSTRUCTIONS
============
## 🚀Execution
Topopt Comec requires the folowing packages on a 3.xx Python environement:

- [Matplotlib](https://pypi.org/project/matplotlib/)
- [Numpy](https://pypi.org/project/numpy/)
- [Scipy](https://pypi.org/project/scipy/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)

To execute it, set the _optimizer2D.py_, _optimizer3D.py_, _topopt_comec.py_ in the same folder and run the _topopt_comec.py_ file.

After launching the program, a defaut case is created. This last is a force inverter, a clasic test case among compliant mechanisms.

## ⚙️Parameters
### Dimensions
The convention used for "dimensions" is classic: (X, Y, Z), but negative values are not accepted. The "volume fraction" coresponds to the filling of the domain and is between 0 and 1.
### ○ Void 
It is possible to impose a "void" area in the domain. This last is defined by a "shape" (square or circle), a "radius" which must be positive and a "center". To disable this functionality, set the "shape" at -.
### → Forces
There must be an "input force" and at least one "output force". Their point of application must me inside the domain. A negative "stiffness" is accepted and has the same effect as flipping the direction. To disable a "force", set the "direction" at -.
### ▲ Supports
There must be at least one "support". They must be inside the domain. To disable a "support", set the "fixed dimensions" parameter at -.
### 🧱 Material
The material is defined by the "Young's modulus" for the elasticity, and the "Poisson's ratio" for the deformation in directions perpendicular to the specific direction of loading. The "Young's modulus" must be between 0 and 10,  and the "Poisson's ratio" must be between 0 and 0.5.
### 💻 Optimizer
The optimisation part is a modified version of the [165 lines topology optimization code](https://www.topopt.mek.dtu.dk/Apps-and-software/Topology-optimization-codes-written-in-Python) from DTU. It constits of a linear optimization with SIMP method. There is one optimizer file for the 2D case and one for the 3D case, but they operates in the exact same way. The first parameter allows the user to chose the "filter" method between density and sensitivity. The second is about the filter's "radius", which must be stricly positive. The "penalization" factor is used in the SIMP method, and 3 is generally a good choice. The time per "iteration" is roughly proportional to the total number of elements. The number of "iteration" must be positive. Before creating a mechanism with many iterations, it is recomended to visualy check the inputs by running 0 "iteration". The iteration process is updated in the terminal. It is possible to save the result by checking the box "save". This will create a folder named _Results_, and place inside the image output as a png file. The box "save" must be checked before clicking "create".

✔️ TEST CASES
==========
## 2D Force inverter
![2d_fi](https://user-images.githubusercontent.com/52052772/139708562-175c7c7b-517a-4c13-b03e-8726c7122669.png)
## 2D Gripper
![2d_gr](https://user-images.githubusercontent.com/52052772/139708570-3eac1db0-dd92-4943-9734-eedd88125791.png)
## 3D force inverter
![3d_fi](https://user-images.githubusercontent.com/52052772/139727931-4a084a13-9b77-44cb-a4f0-3017851fe10a.PNG)

Now it is your turn to create cool mechanisms !

Thank you for using Topopt Comec 🙂

> Just optimize !
