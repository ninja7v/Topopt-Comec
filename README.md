TopoptComec
============

**Interactive topology optimization for compliant mechanisms — unleash your creativity!**

![topoptcomec_demo](https://github.com/user-attachments/assets/61474237-eb3a-48af-ae03-371615038686)

## What is TopoptComec?
TopoptComec helps you design compliant mechanisms — flexible structures that achieve motion through material deformation rather than joints.

Simply draw your domain, set forces and supports, choose your material and optimizer, then watch the algorithm sculpt the optimal shape — in 2D or 3D.

## Why using TopoptComec?
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
git clone https://github.com/ninja7V/topoptcomec.git
cd topoptcomec
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
py main.py -p ForceInverter_2Sup_2D
```
### Create
Tweak the parameters or choose a preset and hit "Create"!

### Export
Once you’re happy with your mechanism, export it for visualization in ParaView or for refinement in your favorite CAD software.

## 📖Wiki
The interface should feel intuitive, but you’ll find detailed visual explanations in the [Wiki](https://github.com/ninja7v/TopoptComec/wiki).

![topoptcomec_intro](https://github.com/user-attachments/assets/a8dab27a-19f7-45c7-850d-23df27416b33)

## ✍️Contribute
Ideas, bug reports, or pull requests — all are welcome. Let’s build something awesome together!

See the CONTRIBUTING.md file for details. 

Thank you for using TopoptComec 🙂

> Just optimize!

## Licensing
This project is licensed under the MIT License - see the LICENSE.txt file for details.

This project also uses the PySide6 library, which is licensed under the GNU Lesser General Public License v3.0 (LGPLv3). The source code for PySide6 can be obtained from its official repository: [https://github.com/pyside/pyside-setup](https://github.com/pyside/pyside-setup).
