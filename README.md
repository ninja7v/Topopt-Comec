Topopt Comec
============

**Interactive topology optimization for compliant mechanisms — unleash your creativity!**

![topopt-comec_demo](https://github.com/user-attachments/assets/6ed953b6-ae72-4844-8c35-a1d284d7478f)

## What is Topopt Comec?
Topopt Comec helps you design compliant mechanisms — flexible structures that achieve motion through material deformation rather than joints.

Simply draw your domain, set forces and supports, choose your material and optimizer, then watch the algorithm sculpt the optimal shape — in 2D or 3D.

## Why using Topopt Comec?
None of the existing competing solutions combine all of these features in a single tool:
- 🔨 **Rigid structure *and* compliant mechanism**  — Same solver, just change the loads.
- 🧊 **2D *and* 3D support** — Real engineering isn’t flat.
- 🚀 **Fast, like really fast** — Designed for performance, not academic demos.
- 🧪 **Flexible** — Tons of parameters to tweak → infinite design possibilities.
- 🛡️ **Reliable** — Bugs fear this program.
- 🍰 **Easy to use** — Intuitive GUI *and* CLI, piece of cake to use.
- 🔓 **Open source** — Transparent, extensible. No black boxes.

## 🚀Quick Start
### Clone/Download the repo
```cmd
git clone https://github.com/ninja7V/topopt-comec.git
cd topopt-comec
```
### Install dependencies
```cmd
pip install -r requirements.txt
```
### Run
GUI:
```cmd
py main.py
```
CLI:
```cmd
py main.py -preset ForceInverter_2Sup_2D
```
### Create
Tweak the parameters or choose a preset and hit "Create"!

### Export
Once you’re happy with your mechanism, export it for visualization in ParaView or for refinement in your favorite CAD software.

## 📖Wiki
The interface should feel intuitive, but you’ll find detailed visual explanations in the [Wiki](https://github.com/ninja7v/Topopt-Comec/wiki).

![topopt-comec_intro](https://github.com/user-attachments/assets/a8dab27a-19f7-45c7-850d-23df27416b33)

## ✍️Contribute
Ideas, bug reports, or pull requests — all are welcome. Let’s build something awesome together!

Now it's your turn to create cool mechanisms.

Thank you for using Topopt Comec 🙂

> Just optimize!

## Licensing
This project is licensed under the MIT License - see the LICENSE.txt file for details.

This project also uses the PySide6 library, which is licensed under the GNU Lesser General Public License v3.0 (LGPLv3). The source code for PySide6 can be obtained from its official repository: [https://github.com/pyside/pyside-setup](https://github.com/pyside/pyside-setup).
