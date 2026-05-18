import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spence
from tqdm.auto import tqdm


class Magnons:

    def __init__(self,L1,L2,a=1,d=5.72e-10):
        self.a1,self.a2 = a* np.array([1,0]),a* np.array([np.cos(np.pi/3),np.sin(np.pi/3)])
        self.b1 = 2 / (np.sqrt(3) * a) * np.array([np.cos(np.pi / 6), -np.sin(np.pi / 6)])
        self.b2 = 2 / (np.sqrt(3) * a) * np.array([0,1])

        self.d = d
        k_pts = []
        for i in range(L1):
            for j in range(L2):
                k = (2*np.pi * i / L1) * self.b1 + (2*np.pi * j / L2) * self.b2
                k_pts.append(k)
        k_pts = np.array(k_pts)
        
        self.kpts = k_pts


        # Fundamental constants
        self.kB = 1.380649e-23
        self.hbar = 1.054571817e-34
    
    def plotKpts(self):
        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot the parallelogram (BZ) edges
        corner = np.array([[0, 0], self.b1, self.b1 + self.b2, self.b2, [0, 0]])
        ax.plot(corner[:, 0], corner[:, 1], 'k--', linewidth=1, label='BZ boundary')

        # Plot lattice points in the BZ
        ax.scatter(self.kpts[:, 0], self.kpts[:, 1], s=8, color='C0', alpha=0.8, label='k-points')

        # Plot reciprocal lattice vectors
        ax.arrow(0, 0, self.b1[0], self.b1[1], head_width=0.3, head_length=0.3, fc='r', ec='r', length_includes_head=True, label=r'$\mathbf{b}_1$')
        ax.arrow(0, 0, self.b2[0], self.b2[1], head_width=0.3, head_length=0.3, fc='g', ec='g', length_includes_head=True, label=r'$\mathbf{b}_2$')

        # Formatting
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        ax.set_title('First Brillouin Zone & Discretization')
        ax.axis('equal')
        ax.legend(['BZ boundary', 'k-points', r'$\mathbf{b}_1$', r'$\mathbf{b}_2$'], loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Takes (0,0,1), and maps to a vector with \phi (in xy plane) and \theta (polar from z axis) spherical coordinates.
    @staticmethod
    def Rot(theta,phi):
        return np.array([[np.cos(theta)*np.cos(phi), - np.sin(phi), np.sin(theta)*np.cos(phi)],
                        [np.cos(theta)*np.sin(phi), np.cos(phi), np.sin(theta)*np.sin(phi)],
                        [-np.sin(theta), 0, np.cos(theta)]])

    @staticmethod
    def Hmats(K,Gamma,GammaPrime):    
        H_x = np.array([[K,GammaPrime,GammaPrime],
                        [GammaPrime,0,Gamma],
                        [GammaPrime,Gamma,0]])
        H_y = np.array([[0,GammaPrime,Gamma],
                        [GammaPrime,K,GammaPrime],
                        [Gamma,GammaPrime,0]])
        H_z = np.array([[0,Gamma,GammaPrime],
                        [Gamma,0,GammaPrime],
                        [GammaPrime,GammaPrime,K]])
        return H_x,H_y,H_z

    @staticmethod
    def HTildeMats(K,Gamma,GammaPrime,theta,phi):

        H_x,H_y,H_z = Magnons.Hmats(K,Gamma,GammaPrime)
        R = Magnons.Rot(theta,phi)
        Htilde_x = R.T @ H_x @ R
        Htilde_y = R.T @ H_y @ R
        Htilde_z = R.T @ H_z @ R
        return Htilde_x,Htilde_y,Htilde_z

    def deltaLambdas(self):
        delta_x = 0*self.a1 -self.a2 
        delta_y = self.a1 - self.a2
        delta_z = 0*self.a1 + 0*self.a2
        return delta_x,delta_y,delta_z

    # get_grad: 0 for original D_k, 1 for kx partial, 2 for ky partial

    def getDk(self,hRatio,K,Gamma,GammaPrime,theta,phi,get_grad=0):
        h = hRatio * np.abs(K)
        if get_grad == 1 or get_grad == 2: h = 0 # derivative of constant is 0

        Htildes = Magnons.HTildeMats(K,Gamma,GammaPrime,theta,phi)
        deltas = self.deltaLambdas()
        zipped_iterable =  list(zip(Htildes, deltas))       

        # k is a (x,y) vector in reciprocal space
        def Dk(k):
            # Construct A_k and B_k
            A_k = np.array([[h,0],[0,h]],dtype=np.complex128)
            A_negk = np.array([[h,0],[0,h]],dtype=np.complex128)
            B_k = np.array([[0,0],[0,0]],dtype=np.complex128)
            B_negk = np.array([[0,0],[0,0]],dtype=np.complex128)
            for Htilde, delta in zipped_iterable:
                h11,h22,h33,h12 = Htilde[0,0],Htilde[1,1],Htilde[2,2], Htilde[0,1]
                exponent_pos = np.exp(1j * np.dot(k,delta))
                exponent_neg = np.exp(-1j * np.dot(k,delta))

                if get_grad == 1:
                    exponent_pos *= 1j * delta[0]
                    exponent_neg *= -1j * delta[0]
                    h33 = 0 # derivative of constant is 0
                elif get_grad == 2:
                    exponent_pos *= 1j * delta[1]
                    exponent_neg *= -1j * delta[1]
                    h33 = 0 # derivative of constant is 0
                

                next_term_A_k = np.array([[-h33, 1/2 * (h11 + h22) * exponent_pos],
                                        [1/2 * (h11 + h22) * exponent_neg, -h33]],dtype=np.complex128)
                
                next_term_B_k = 1/2 * (h11 + 2j * h12 - h22) * np.array([[0,exponent_pos],
                                                                        [exponent_neg,0]],dtype=np.complex128)
                
                next_term_A_negk = np.array([[-h33, 1/2 * (h11 + h22) * exponent_neg],
                                        [1/2 * (h11 + h22) * exponent_pos, -h33]],dtype=np.complex128)
                
                next_term_B_negk = 1/2 * (h11 + 2j * h12 - h22) * np.array([[0,exponent_neg],
                                                                        [exponent_pos,0]],dtype=np.complex128)
                
                A_k += next_term_A_k
                B_k += next_term_B_k
                A_negk += next_term_A_negk
                B_negk += next_term_B_negk
            

            # Construct D_k block diagonals with A_k, and off block diagonasl with B_k
            D_k = np.block([[A_k,B_k],[B_negk.conj(),A_negk.T]])

            return D_k
        
        return Dk


    # Returns two arrays: eigenvalues and eigenvectors, in that order
    # Eigenvalues is 1D array, two values, which are the sorted eigenvalues
    # Eigenvectors is 2D, 4x2 array, each column a 4-element vector for the corresponding eigenvalue
    # Will get all 4 Eigenvalues if set getall=True 2 should be negative, 2 should be positive
    @staticmethod
    def getDkEigs(Dk,k,getall=False):
        sigma3 = np.diag([1, 1, -1, -1])
        vals, vecs = np.linalg.eig(sigma3 @ Dk(k))
        vals = np.real_if_close(vals, tol=1000)
        vals_real = np.real(vals)

        if getall:
            eigpos_idx = np.arange(4)
        else:
            eigpos_idx = np.where(vals_real > 1e-10)[0]
            if eigpos_idx.size != 2:
                raise RuntimeError(f"Expected 2 positive magnon modes at k={k}, got {vals_real[eigpos_idx]}")

        order = np.argsort(vals_real[eigpos_idx])
        sel = eigpos_idx[order]
        final_vals, final_vecs = vals_real[sel], vecs[:, sel].astype(np.complex128, copy=True)

        # Paraunitary normalization with sigma3 metric: v^dagger sigma3 v = sign(eigenvalue)
        for j in range(final_vecs.shape[1]):
            v = final_vecs[:, j]
            metric_norm = np.real_if_close(np.conj(v).T @ sigma3 @ v, tol=1000)
            metric_norm = np.real(metric_norm)
            if np.abs(metric_norm) < 1e-12:
                raise RuntimeError(f"Near-zero sigma3 norm eigenvector at k={k}, band index={sel[j]}")

            v = v / np.sqrt(np.abs(metric_norm))

            final_vecs[:, j] = v

        return final_vals, final_vecs

    def plot(self,func1,func2,cmap="viridis",hexcolor="white"):

        # Optional plotting grid for Figure 3-style dispersion maps
        # Cartesian reciprocal-space grid with axes shown as k_x a and k_y a in [-2π, 2π]
        Nkx, Nky = 161, 161
        kx_a_vals = np.linspace(-2 * np.pi, 2 * np.pi, Nkx)
        ky_a_vals = np.linspace(-2 * np.pi, 2 * np.pi, Nky)
        KX_a, KY_a = np.meshgrid(kx_a_vals, ky_a_vals, indexing='xy')

        # Dk expects k in units of 1/a
        plotkpts = np.column_stack([(KX_a / a).ravel(), (KY_a / a).ravel()])

        fig3_ticks = np.array([-2 * np.pi, -np.pi, 0.0, np.pi, 2 * np.pi])
        fig3_ticklabels = [r'$-2\pi$', r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$']

        vals1,vals2 = np.zeros(len(plotkpts), dtype=np.float64),np.zeros(len(plotkpts), dtype=np.float64)
        for i, k in enumerate(plotkpts):
            val1,val2 = func1(k),func2(k)
            vals1[i],vals2[i] = val1,val2
        
        en1 = vals1.reshape(Nky, Nkx)
        en2 = vals2.reshape(Nky, Nkx)

        # First-BZ hexagon from reciprocal primitive vectors B1=2π b1, B2=2π b2
        B1 = 2 * np.pi * self.b1
        B2 = 2 * np.pi * self.b2
        hex_vertices = np.array([
            (2 * B1 + B2) / 3,
            (B1 + 2 * B2) / 3,
            (-B1 + B2) / 3,
            -(2 * B1 + B2) / 3,
            -(B1 + 2 * B2) / 3,
            (B1 - B2) / 3,
        ])
        hex_plot = np.vstack([hex_vertices, hex_vertices[0]]) * a  # axes are k_x a, k_y a

        vmin = min(np.min(en1), np.min(en2))
        vmax = max(np.max(en1), np.max(en2))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
        im0 = axes[0].pcolormesh(KX_a, KY_a, en1, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].pcolormesh(KX_a, KY_a, en2, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)

        for ax, n in zip(axes, [1, 2]):
            ax.plot(hex_plot[:, 0], hex_plot[:, 1], color=hexcolor, linewidth=1.8)
            ax.set_xlim(-2 * np.pi, 2 * np.pi)
            ax.set_ylim(-2 * np.pi, 2 * np.pi)
            ax.set_xticks(fig3_ticks, fig3_ticklabels)
            ax.set_yticks(fig3_ticks, fig3_ticklabels)
            ax.set_xlabel(r'$k_x a$')
            ax.set_ylabel(r'$k_y a$')
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(rf'n = {n}')

        cbar = fig.colorbar(im0, ax=axes, shrink=0.92)
        cbar.set_label(r'$\epsilon_{n\mathbf{k}}/S|K|$')
        plt.show()

        return vals1,vals2

    @staticmethod
    def berryCurvatureZ(n,k,Dk,gradx_Dk,grady_Dk):
        sign_metric  = np.diag([-1, -1, 1, 1]) # Not strictly sigma3, need the negatives to pair with the negative eigenvalues/holes, and are in front since you sorted them. 
        eigvals,eigvecs = Magnons.getDkEigs(Dk,k,True)
        totalBands = eigvals.size

        omega_nk = 0
        for m in range(totalBands):
            if m == n:
                continue
            else:
                numerator = (np.conj(eigvecs[:,n].T) @ gradx_Dk(k) @ eigvecs[:,m]) * (np.conj(eigvecs[:,m].T) @ grady_Dk(k) @ eigvecs[:,n])
                denominator = (eigvals[n] - eigvals[m]) ** 2
                sign = sign_metric[m][m]
                omega_nk += numerator / denominator * sign

        omega_nk = -2 * np.imag(omega_nk)
        return omega_nk

    @staticmethod
    def c2(x):
        return (1+x) * (np.log((1+x)/x))**2 - np.log(x)**2 - 2*spence(1+x)

    @staticmethod
    def g(x,T,kB):
        return 1 / (np.exp(x/(kB*T) )-1)

    @staticmethod
    def singlek(e_nk,omega_nk,T,kB):
        return (Magnons.c2(Magnons.g(e_nk,T,kB)) - np.pi**2/3) * omega_nk

    def thermallhallconductivity(self,hRatio,T,K,Gamma,GammaPrime,theta,phi):
        hbar = 1.054571817e-34
        kB = 1.380649e-23
        d = 5.72e-10


        area2d = np.abs(self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0])
        V = d * len(self.kpts) * area2d
        deltaA_k = (2 * np.pi) ** 2 / (len(self.kpts) * area2d)

        Dk = self.getDk(hRatio,K,Gamma,GammaPrime,theta,phi)
        gradx_Dk = self.getDk(hRatio,K,Gamma,GammaPrime,theta,phi,1)
        grady_Dk = self.getDk(hRatio,K,Gamma,GammaPrime,theta,phi,2)
        
        k_iter = tqdm(self.kpts, total=len(self.kpts), desc="k-point loop")

        kappa_xy = 0
        chern_numbers = [0,0,0,0]
        for kx,ky in k_iter:
            k = (kx,ky)
            e_nks = Magnons.getDkEigs(Dk,k,True)[0]
            for i in range(len(e_nks)):
                enk = e_nks[i]
                enk *= 40 * kB
                if enk > 0:
                    omega_nk = Magnons.berryCurvatureZ(i,k,Dk,gradx_Dk,grady_Dk)
                    kappa_xy += Magnons.singlek(enk,omega_nk,T,kB)
                    chern_numbers[i] += omega_nk * deltaA_k

        kappa_xy *= - kB**2 * T / (V * hbar)

        chern_numbers = np.array(chern_numbers) / (2 * np.pi) 

        return kappa_xy, chern_numbers
    
    @staticmethod
    def plotFig2Style(x,y,color="green",loc="lower right",hRatio=0.1,title=""):
        fig, ax = plt.subplots(figsize=(4, 2.6))
        ax.plot(x, y, "o-", color=color, markersize=4, linewidth=1.8, label=rf"$h/S|K| = {hRatio}$")

        ax.set_xlim(0, 20)
        ax.set_xlabel(r"$T\,(\mathrm{K})$")
        ax.set_ylabel(r"$\kappa_{xy}/T\; (\mathrm{mW\,K^{-2}\,m^{-1}})$")
        ax.tick_params(direction="in")
        ax.legend(loc=loc, frameon=True, fontsize=8)
        ax.set_title(title)

        # Right axis used in Fig. 2: kappa_xy^{2D}/T normalized by (pi k_B^2 / 6 hbar)
        hbar = 1.054571817e-34
        d,kB = 5.72e-10,1.380649e-23
        scale = (1e-3 * d) / (np.pi * kB**2 / (6 * hbar))
        secax = ax.secondary_yaxis(
            "right",
            functions=(lambda y: y * scale, lambda y: y / scale),
        )
        secax.set_ylabel(r"$\kappa_{xy}^{2D}/T\; (\pi k_B^2/6\hbar)$")
        secax.tick_params(direction="in")

        fig.tight_layout()
        plt.show()