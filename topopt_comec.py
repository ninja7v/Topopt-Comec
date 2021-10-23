# A 350 LINE GUI CODE FOR TOPOLOGY OPTIMIZATION BY LUC PREVOST, 2021
from __future__ import division
import os
from matplotlib import colors
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import optimizer2D, optimizer3D

def main():
	root = Tk()
	gui = Window(root)
	gui.root.mainloop()
	return None

class Window:
	def __init__(self, root):
		self.root = root
		# Windows title
		self.root.title("Topopt Comec")
		# Windows size
		self.root.geometry('1225x725')
		# Initialization
		(self.nelx, self.nely, self.nelz) = (70, 50, 0)
		self.volfrac = 0.3
		(self.cx, self.cy, self.cz, self.r, self.v) = (0, 0, 0, 0, '-')
		(self.fix, self.fiy, self.fiz, self.a1) = (0, 25, 0, '→')
		(self.fo1x, self.fo1y, self.fo1z, self.a2) = (70, 25, 0, '←')
		(self.fo2x, self.fo2y, self.fo2z, self.a3) = (0, 0, 0, '-')
		(self.s1x, self.s1y, self.s1z, self.d1) = (0, 0, 0, 'XYZ')
		(self.s2x, self.s2y, self.s2z, self.d2) = (0, 50, 0, 'XYZ')
		(self.s3x, self.s3y, self.s3z, self.d3) = (0, 0, 0, '-')
		(self.s4x, self.s4y, self.s4z, self.d4) = (0, 0, 0, '-')
		(self.E, self.nu) = (1, 0.25)
		self.save_results = 0
		# Title
		Label(self.root, text = "Linear topology optimization for compliant mechanisms", font=("Arial", 20), fg = 'black').grid(row=0, column=0, columnspan = 15)
		# Dimensions (1-3)
		Label(self.root, text = "Dimensions", font=("Arial", 12)).grid(row=1, column=20, columnspan = 5) 
		self.nelx_entry = Entry(self.root, width = 5)
		self.nelx_entry.grid(row=2, column = 20)
		self.nelx_entry.insert(0, self.nelx)
		Label(self.root, text = "x").grid(row=2, column=21)
		self.nely_entry = Entry(self.root, width = 5)
		self.nely_entry.grid(row=2, column = 22)
		self.nely_entry.insert(0, self.nely)
		Label(self.root, text = "x").grid(row=2, column=23)
		self.nelz_entry = Entry(self.root, width = 5)
		self.nelz_entry.grid(row=2, column = 24)
		self.nelz_entry.insert(0, self.nelz)
		Label(self.root, text = "Volume fraction").grid(row=3, column=20, columnspan = 3)
		self.volfrac_entry = Entry(self.root, width = 5)
		self.volfrac_entry.grid(row=3, column = 23)
		self.volfrac_entry.insert(0, self.volfrac)
		# Void area (4-6)
		shapes = ['-', '□', '○']
		Label(self.root, text = "Void area", font=("Arial", 12)).grid(row=4, column=20, columnspan = 5)
		Label(self.root, text = "Shape").grid(row=5, column=20)
		CB1 = ttk.Combobox(self.root, values=shapes, width = 5, textvariable = self.v)
		CB1.set(self.v)
		CB1.grid(row = 5, column = 21)
		Label(self.root, text = "Radius").grid(row=5, column=23)
		self.fox_entry = Entry(self.root, width = 5)
		self.fox_entry.grid(row=5, column = 24)
		self.fox_entry.insert(0, self.cx)
		Label(self.root, text = "Center").grid(row=6, column=20)
		self.fox_entry = Entry(self.root, width = 5)
		self.fox_entry.grid(row=6, column = 21)
		self.fox_entry.insert(0, self.cx)
		self.foy_entry = Entry(self.root, width = 5)
		self.foy_entry.grid(row=6, column = 22)
		self.foy_entry.insert(0, self.cy)
		self.foz_entry = Entry(self.root, width = 5)
		self.foz_entry.grid(row=6, column = 23)
		self.foz_entry.insert(0, self.cz)
		# Force location (7-10)
		arrows = ['-', '↑', '↓', '→', '←']
		Label(self.root, text = "Forces location", font=("Arial", 12)).grid(row=7, column=20, columnspan = 5)
		Label(self.root, text = "In").grid(row=8, column=20)
		self.fix_entry = Entry(self.root, width = 5)
		self.fix_entry.grid(row=8, column = 21)
		self.fix_entry.insert(0, self.fix)
		self.fiy_entry = Entry(self.root, width = 5)
		self.fiy_entry.grid(row=8, column = 22)
		self.fiy_entry.insert(0, self.fiy)
		self.fiz_entry = Entry(self.root, width = 5)
		self.fiz_entry.grid(row=8, column = 23)
		self.fiz_entry.insert(0, self.fiz)
		self.a1_entry = tk.StringVar() 
		CB2 = ttk.Combobox(self.root, values=arrows, width = 5, textvariable = self.a1)
		CB2.set(self.a1)
		CB2.grid(row = 8, column = 24)
		Label(self.root, text = "Out1").grid(row=9, column=20)
		self.fox_entry = Entry(self.root, width = 5)
		self.fox_entry.grid(row=9, column = 21)
		self.fox_entry.insert(0, self.fo1x)
		self.foy_entry = Entry(self.root, width = 5)
		self.foy_entry.grid(row=9, column = 22)
		self.foy_entry.insert(0, self.fo1y)
		self.foz_entry = Entry(self.root, width = 5)
		self.foz_entry.grid(row=9, column = 23)
		self.foz_entry.insert(0, self.fo1z)
		CB3 = ttk.Combobox(self.root, values=arrows, width = 5, textvariable = self.a2)
		CB3.set(self.a2)
		CB3.grid(row = 9, column = 24)
		Label(self.root, text = "Out2").grid(row=10, column=20)
		self.fox_entry = Entry(self.root, width = 5)
		self.fox_entry.grid(row=10, column = 21)
		self.fox_entry.insert(0, self.fo2x)
		self.foy_entry = Entry(self.root, width = 5)
		self.foy_entry.grid(row=10, column = 22)
		self.foy_entry.insert(0, self.fo2y)
		self.foz_entry = Entry(self.root, width = 5)
		self.foz_entry.grid(row=10, column = 23)
		self.foz_entry.insert(0, self.fo2z)
		CB4 = ttk.Combobox(self.root, values=arrows, width = 5, textvariable = self.a3)
		CB4.set(self.a2)
		CB4.grid(row = 10, column = 24)
		# Supports location (11-15)
		dim = ['-', 'X', 'Y', 'Z', 'XYZ']
		Label(self.root, text = "Supports location", font=("Arial", 12)).grid(row=11, column=20, columnspan = 5)
		Label(self.root, text = "S1").grid(row=12, column=20)
		self.s1x_entry = Entry(self.root, width = 5)
		self.s1x_entry.grid(row=12, column = 21)
		self.s1x_entry.insert(0, self.s1x)
		self.s1y_entry = Entry(self.root, width = 5)
		self.s1y_entry.grid(row=12, column = 22)
		self.s1y_entry.insert(0, self.s1y)
		self.s1z_entry = Entry(self.root, width = 5)
		self.s1z_entry.grid(row=12, column = 23)
		self.s1z_entry.insert(0, self.s1z)
		CB5 = ttk.Combobox(self.root, values=dim, width = 5, textvariable = self.d1)
		CB5.set(self.d1)
		CB5.grid(row = 12, column = 24)
		Label(self.root, text = "S2").grid(row=13, column=20)
		self.s2x_entry = Entry(self.root, width = 5)
		self.s2x_entry.grid(row=13, column = 21)
		self.s2x_entry.insert(0, self.s2x)
		self.s2y_entry = Entry(self.root, width = 5)
		self.s2y_entry.grid(row=13, column = 22)
		self.s2y_entry.insert(0, self.s2y)
		self.s2z_entry = Entry(self.root, width = 5)
		self.s2z_entry.grid(row=13, column = 23)
		self.s2z_entry.insert(0, self.s2z)
		CB6 = ttk.Combobox(self.root, values=dim, width = 5, textvariable = self.d2)
		CB6.set(self.d2)
		CB6.grid(row = 13, column = 24)
		Label(self.root, text = "S3").grid(row=14, column=20)
		self.s2x_entry = Entry(self.root, width = 5)
		self.s2x_entry.grid(row=14, column = 21)
		self.s2x_entry.insert(0, self.s3x)
		self.s2y_entry = Entry(self.root, width = 5)
		self.s2y_entry.grid(row=14, column = 22)
		self.s2y_entry.insert(0, self.s3y)
		self.s2z_entry = Entry(self.root, width = 5)
		self.s2z_entry.grid(row=14, column = 23)
		self.s2z_entry.insert(0, self.s3z)
		CB7 = ttk.Combobox(self.root, values=dim, width = 5, textvariable = self.d3)
		CB7.set(self.d3)
		CB7.grid(row = 14, column = 24)
		Label(self.root, text = "S4").grid(row=15, column=20)
		self.s2x_entry = Entry(self.root, width = 5)
		self.s2x_entry.grid(row=15, column = 21)
		self.s2x_entry.insert(0, self.s4x)
		self.s2y_entry = Entry(self.root, width = 5)
		self.s2y_entry.grid(row=15, column = 22)
		self.s2y_entry.insert(0, self.s4y)
		self.s2z_entry = Entry(self.root, width = 5)
		self.s2z_entry.grid(row=15, column = 23)
		self.s2z_entry.insert(0, self.s4z)
		CB8 = ttk.Combobox(self.root, values=dim, width = 5, textvariable = self.d4)
		CB8.set(self.d4)
		CB8.grid(row = 15, column = 24)
		# Material (16-18)
		Label(self.root, text = "Material", font=("Arial", 12)).grid(row=16, column=20, columnspan = 5)
		Label(self.root, text = "Young's modulus").grid(row=17, column=20, columnspan = 3)
		self.E_entry = Entry(self.root, width = 5)
		self.E_entry.grid(row=17, column = 23)
		self.E_entry.insert(0, self.E)
		Label(self.root, text = "N/m\N{superscript two}").grid(row=17, column=24)
		Label(self.root, text = "Poisson's ratio").grid(row=18, column=20, columnspan = 3)
		self.nu_entry = Entry(self.root, width = 5)
		self.nu_entry.grid(row=18, column = 23)
		self.nu_entry.insert(0, self.nu)
		# Save results (19)
		self.save_result_entry = IntVar()
		C = Checkbutton(self.root, text = "Save results", font=("Arial", 12), variable = self.save_result_entry, onvalue = 1, offvalue = 0, height=1, width = 20).grid(row=19, column=20, columnspan = 5)
		# Compute (20)
		B = Button(self.root, text="Calculate", font=("Arial", 12), command = self.update_values)
		B.grid(row=20, column=20, columnspan = 5)
		self.root.bind("<Return>", self.update_values)
		self.plot_values_2D(optimizer2D.optimize(self.nelx, self.nely, self.volfrac, self.fix, self.fiy, self.a1, 
			self.fo1x, self.fo1y, self.a2, self.fo2x, self.fo2y, self.a3, self.s1x, self.s1y, self.d1, self.s2x, self.s2y, self.d2, self.E, self.nu))
		pass
	
	def update_values(self, event=None):
		(self.nelx, self.nely, self.nelz) = (int(self.nelx_entry.get()), int(self.nely_entry.get()), int(self.nelz_entry.get()))
		self.volfrac = float(self.volfrac_entry.get())
		(self.fix, self.fiy, self.fiz, self.a1) = (int(self.fix_entry.get()), int(self.fiy_entry.get()), int(self.fiz_entry.get()), str(self.a1))
		(self.fox, self.foy, self.foz, self.a2) = (int(self.fox_entry.get()), int(self.foy_entry.get()), int(self.foz_entry.get()), str(self.a2))
		(self.s1x, self.s1y, self.s1z, self.d1) = (int(self.s1x_entry.get()), int(self.s1y_entry.get()), int(self.s1z_entry.get()), str(self.d1))
		(self.s2x, self.s2y, self.s2z, self.d2) = (int(self.s2x_entry.get()), int(self.s2y_entry.get()), int(self.s2z_entry.get()), str(self.d2))
		(self.E, self.nu) = (float(self.E_entry.get()), float(self.nu_entry.get()))
		self.save_results = int(self.save_result_entry.get())
		if self.check_entry():
			if self.nelz==0:
				self.plot_values_2D(optimizer2D.optimize(self.nelx, self.nely, self.volfrac, self.fix, self.fiy, self.a1, 
						self.fox, self.foy, self.a2, self.s1x, self.s1y, self.d1, self.s2x, self.s2y, self.d2, self.E, self.nu))
			if self.nelz>0:
				self.plot_values_3D(optimizer3D.optimize(self.nelx, self.nely, self.nelz, self.volfrac, self.fix, self.fiy, self.fiz, self.a1, 
						self.fox, self.foy, self.foz, self.a2, self.s1x, self.s1y, self.s1z, self.d1, self.s2x, self.s2y, self.s2z, self.d2, self.E, self.nu))
		return None
	
	def check_entry(self, ):
		valid_entry = False
		# Dimensions
		if self.nelx>200: messagebox.showerror("Dimensions input error", "The number of element in X is superior to 200.")
		elif self.nelx<0: messagebox.showerror("Dimensions input error", "The number of element in X is inferior to 0.")
		elif self.nely>200: messagebox.showerror("Dimensions input error", "The number of element in Y is superior to 200.")
		elif self.nely<0: messagebox.showerror("Dimensions input error", "The number of element in Y is inferior to 0.")
		elif self.nelz>200: messagebox.showerror("Dimensions input error", "The number of element in Z is superior to 200.")
		elif self.nelz<0: messagebox.showerror("Dimensions input error", "The number of element in Z is inferiorto 0.")
		elif self.volfrac>1: messagebox.showerror("Dimensions input error", "The volume fraction is upper than 1.")
		elif self.volfrac<0: messagebox.showerror("Dimensions input error", "The volume fraction is lower than 0.")
		# Void area
		if self.r<0: messagebox.showerror("Void area input error", "The radius is lower than 0.")
		# Forces location
		elif self.fix>self.nelx: messagebox.showerror("Forces input error", "The input force is outside the domain.")
		elif self.fix<0: messagebox.showerror("Forces input error", "The input force is outside the domain.")
		elif self.fiy>self.nely: messagebox.showerror("Forces input error", "The input force is outside the domain.")
		elif self.fiy<0: messagebox.showerror("Forces input error", "The input force is outside the domain.")
		elif self.fiz>self.nelz: messagebox.showerror("Forces input error", "The input force is outside the domain.")
		elif self.fiz<0: messagebox.showerror("Forces input error", "The input force is outside the domain.")
		elif self.fox>self.nelx: messagebox.showerror("Forces input error", "The output force is outside the domain.")
		elif self.fox<0: messagebox.showerror("Forces input error", "The output force is outside the domain.")
		elif self.foy>self.nely: messagebox.showerror("Forces input error", "The output force is outside the domain.")
		elif self.foy<0: messagebox.showerror("Forces input error", "The output force is outside the domain.")
		elif self.foz>self.nelz: messagebox.showerror("Forces input error", "The output force is outside the domain.")
		elif self.foz<0: messagebox.showerror("Forces input error", "The output force is outside the domain.")
		# Supports location
		elif self.s1x>self.nelx: messagebox.showerror("Supports input error", "The support 1 is outside the domain.")
		elif self.s1x<0: messagebox.showerror("Supports input error", "The support 1 is outside the domain.")
		elif self.s1y>self.nely: messagebox.showerror("Supports input error", "The support 1 is outside the domain.")
		elif self.s1y<0: messagebox.showerror("Supports input error", "The support 1 is outside the domain.")
		elif self.s1z>self.nelz: messagebox.showerror("Supports input error", "The support 1 is outside the domain.")
		elif self.s1z<0: messagebox.showerror("Supports input error", "The support 1 is outside the domain.")
		elif self.s2x>self.nelx: messagebox.showerror("Supports input error", "The support 2 is outside the domain.")
		elif self.s2x<0: messagebox.showerror("Supports input error", "The support 2 is outside the domain.")
		elif self.s2y>self.nely: messagebox.showerror("Supports input error", "The support 2 is outside the domain.")
		elif self.s2y<0: messagebox.showerror("Supports input error", "The support 2 is outside the domain.")
		elif self.s2z>self.nelz: messagebox.showerror("Supports input error", "The support 2 is outside the domain.")
		elif self.s2z<0: messagebox.showerror("Supports input error", "The support 2 is outside the domain.")
		elif self.s3x>self.nelx: messagebox.showerror("Supports input error", "The support 3 is outside the domain.")
		elif self.s3x<0: messagebox.showerror("Supports input error", "The support 3 is outside the domain.")
		elif self.s3y>self.nely: messagebox.showerror("Supports input error", "The support 3 is outside the domain.")
		elif self.s3y<0: messagebox.showerror("Supports input error", "The support 3 is outside the domain.")
		elif self.s3z>self.nelz: messagebox.showerror("Supports input error", "The support 3 is outside the domain.")
		elif self.s3z<0: messagebox.showerror("Supports input error", "The support 3 is outside the domain.")
		elif self.s4x>self.nelx: messagebox.showerror("Supports input error", "The support 4 is outside the domain.")
		elif self.s4x<0: messagebox.showerror("Supports input error", "The support 4 is outside the domain.")
		elif self.s4y>self.nely: messagebox.showerror("Supports input error", "The support 4 is outside the domain.")
		elif self.s4y<0: messagebox.showerror("Supports input error", "The support 4 is outside the domain.")
		elif self.s4z>self.nelz: messagebox.showerror("Supports input error", "The support 4 is outside the domain.")
		elif self.s4z<0: messagebox.showerror("Supports input error", "The support 4 is outside the domain.")
		# Material
		elif self.E>1: messagebox.showerror("Material input error", "The Young's modulus is upper than 10.")
		elif self.E<0: messagebox.showerror("Material input error", "The Young's modulus is lower than 0.")
		elif self.nu>1: messagebox.showerror("Material input error", "The Poisson's ratio is upper than 1.")
		elif self.nu<0: messagebox.showerror("Material input error", "The Poisson's ratio is lower than 0.")
		else: valid_entry = True
		return valid_entry

	def plot_values_2D(self, xPhys):
		plt.ion() # Ensure that redrawing is possible
		fig, ax = plt.subplots()
		im = ax.imshow(-xPhys.reshape((self.nelx, self.nely)).T, cmap='gray',\
				 interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
		im.set_array(-xPhys.reshape((self.nelx, self.nely)).T)
# 		plt.quiver((din//2)//(self.nely+1)-0.5, (din//2)%(self.nely+1)-0.5, self.nelx//2, 0, color=['m'], scale=100)
# 		plt.quiver((dout/2)//(self.nely+1)-0.5, (dout/2)%(self.nely+1)-0.5, -self.nelx//2, 0, color=['m'], scale=100)
		chart = FigureCanvasTkAgg(fig, self.root)
		chart.get_tk_widget().grid(row = 1, column = 0, rowspan = 20, columnspan = 20, sticky=E+W+N+S)
		# Save data
		dirname = 'Results'
		if not os.path.exists(dirname):
			os.makedirs(dirname)
			print("Directory " , dirname ,  " Created ")
		if self.save_results:
			fig.savefig('%s.png' % ('force_inverter_2D'), bbox_inches='tight', dpi=200)
		return None

	def plot_values_3D(self, xPhys):
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		x = []
		y = []
		z = []
		for el in range(nel):
			if xPhys[el] > 0.6:
				(xe, ye, ze) = getCoordinates(el, 1, nelx, nely, nelz)
				x.append(xe)
				y.append(ye)
				z.append(ze)
		ax.scatter(x, y, z, s=6000/min(nelx, nely, nelz), marker='s', c='black')
		# Plot forces
		# for i in range(nf):
		#     (xf, yf, zf) = getCoordinates(d[i], 0, nelx, nely, nelz)
		#     if d[i]%3==0:
		#         ax.quiver(xf, yf, zf, np.sign(dVal[i])*nelx//2, 0, 0, linewidths=10, color = ['m'])
		#     elif d[i]%3==1:
		#         ax.quiver(xf, yf, zf, 0, np.sign(dVal[i])*nely//2, 0, linewidths=10, color = ['m'])
		#     elif d[i]%3==2:
		#         ax.quiver(xf, yf, zf, 0, 0, np.sign(dVal[i])*nelz//2, linewidths=10, color = ['m'])
		# Plot supports
		# for i in range(ns):
		#     (xs, ys, zs) = getCoordinates(s[i], 0, nelx, nely, nelz)
		#     ax.scatter([xs], [ys], [zs], s=1000/max(nelx, nely, nelz)+10, marker='s', c='blue')
		ax.set_xlabel("x")
		x.set_ylabel("y")
		ax.set_zlabel("z")
		ax.set_title('3D compliant force inverter\
			\n it.: {0} , Vol.: {1:.3f}, penal.:{2}, rmin:{3}'.format(loop, (g+volfrac*nel)/nel, penal, rmin), pad = 10)
		plt.show()
		# Save data
		dirname = 'Results'
		if not os.path.exists(dirname):
			os.makedirs(dirname)
			print("Directory " , dirname ,  " Created ")
		if self.save_results:
			fig.savefig('%s.png' % ('force_inverter_2D'), bbox_inches='tight', dpi=200)
		return None

	pass

if __name__ == '__main__':
    main()