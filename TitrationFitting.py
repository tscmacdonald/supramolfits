import pandas as pd
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit
import itertools
import matplotlib.cm as cm
import seaborn as sns


#Global fit for 2:1 H:G complex data


def Global21(data):
	'''Fitting function for 2:1 Host:Gueset stoichiometry.
	This function calculates association constants from NMR titration data as a Pandas dataframe with the following columns
	[Host concentration / Molar] [Guest concentration / Molar] [Peak 1 / ppm] [ Peak 2 / ppm]...
	...for an arbitry number of peaks. If desired, this can be called from the clipboard using pd.read_clipboard(), eg:
	params = Global21(pd.read_clipboard())
	
	Binding parameters are calculated using a global nonlinear least-squares fit over all provided peaks.
	'''
    def H21(G0,H0,K1,K2):
		'''H21 calculate the concentration of free host, [H], given G0 (total gues concentration), H0 (total host concentration), K1 (H + G <=> [HG] association constant), and K2 (H + [HG] <=> [H2G] association constant).
		'''
        G0,H0,K1,K2 = np.asarray(G0), np.asarray(H0), np.asarray(K1), np.asarray(K2)
        try:
            length = len(G0)
            coeffs = np.zeros((length,4))
            for i in range(length):
                coeffs[i] = [K1*K2, K1*(2*K2*G0[i] - K2*H0[i] + 1), K1*(G0[i]-H0[i])+1, -H0[i]]
            roots = np.zeros((len(G0),3))
            R = np.zeros(len(G0))
            for i in range(len(G0)):
                roots[i] = np.roots(coeffs[i])
                R[i] = min(roots[i][roots[i] >= 0])
            return R
        except TypeError:
            coeffs = [K1*K2, K1*(2*K2*G0 - K2*H0 + 1), K1*(G0-H0)+1, -H0]
            R =  np.roots(coeffs)
        return min(R[R >= 0])

    def nmr21(X,K1,K2,delHG,delH2G):
		'''nmr21: calculates chemical shift difference delDel for a given peak from the free host chemical shift as a function of X = H0, G0, K1, K2, delHG = chemical shift difference on first association, delH2G = chemical shift difference on second association.'''
        H0, G0 = X
        H = H21(G0,H0,K1,K2)
        delDel = (delHG*K1*G0*H + 2*delH2G*G0*K1*K2*H**2)/(H0*(1+K1*H+K1*K2*H**2))
        return delDel

    def nmr21_dataset(X,params,i):
		'''Given X = H0, G0 and params = lmfit parameters element for multiple peaks, calculates chemical shifts for each peak, for each [G0, H0] pairing'''
        K1 = params['K1_%i' % (i+1)].value
        K2 = params['K2_%i' % (i+1)].value
        delHG = params['delDHG1_%i' % (i+1)].value
        delHG2 = params['delDHG2_%i' % (i+1)].value
        return nmr21(X,K1,K2,delHG,delHG2)

    def objective(params, X, data):
        """ calculate total residual for fits to several data sets"""
        dataT = np.array(data.T[2:])
        ndata, nx = dataT.shape
        resid = 0.0*dataT[:]
        # make residual per data set
        for i in range(ndata):
            dataT[i] = dataT[i] - dataT[i][0]
            resid[i, :] = dataT[i, :] - nmr21_dataset(X,params,i)
        # now flatten this to a 1D array, as minimize() needs
        return resid.flatten()

    X = data.iloc[:,0],data.iloc[:,1] 

    dataT = np.array(data.T[2:])
    fit_params = Parameters()
    for iy, y in enumerate(dataT):
        fit_params.add('K1_%i' % (iy+1), value=1000, min=0.0,  max=1e6)
        fit_params.add('K2_%i' % (iy+1), value=100, min=0.0,  max=1e4)
        fit_params.add('delDHG1_%i' % (iy+1), value=1.0, min=-4.0,  max=4.0)
        fit_params.add('delDHG2_%i' % (iy+1), value=-1.0, min=-2.0,  max=2.0)

    for iy in range(2,len(dataT)+1):
        fit_params['K1_%i' % iy].expr='K1_1'
        fit_params['K2_%i' % iy].expr='K2_1'
    
    
    result = minimize(objective, fit_params, args=(X, data))
    #report_fit(result.params)
    
    palette = itertools.cycle(sns.color_palette())
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True)
    plt.rcParams["lines.markeredgewidth"] = 1.5
    interp = np.linspace(min(X[1]),max(X[1]),1000)
    Xinterp = (interp*0.0+X[0][0]), interp
    K1, K2 = result.params['K1_1'].value, result.params['K2_1'].value
    Hinterp = H21(Xinterp[1],Xinterp[0],K1,K2)
    
    Ginterp = (-K1+np.sqrt(K1**2 - 8*K1*K2*(1-Xinterp[0]/Hinterp)))/(4*K1*K2)
    
    HGinterp = K1*Hinterp*Ginterp
    H2Ginterp = K2*Ginterp*HGinterp
    
    
    for i in range(len(dataT)):
        c = next(palette)
        y_fit = nmr21_dataset(X, result.params, i)
        Yinterp = nmr21_dataset(Xinterp, result.params, i)
        ax1.plot(X[1]/X[0], dataT[i, :]-dataT[i,0],'o',color=c,markerfacecolor='None')
        ax1.plot(Xinterp[1]/Xinterp[0], Yinterp, '-', color=c)
        ax2.plot(X[1]/X[0],dataT[i,:]-dataT[i,0]-y_fit, 'o--',color=c,markerfacecolor='None')
        ax1.set_ylabel('Δδ')
        ax2.set_ylabel('Residual')
        
        ax2.axhline(y=0,color='k')
    ax3.plot(Xinterp[1]/Xinterp[0], Hinterp/Xinterp[0],label='[H]')
    ax3.plot(Xinterp[1]/Xinterp[0], HGinterp/Xinterp[0],label='[HG]')
    ax3.plot(Xinterp[1]/Xinterp[0], H2Ginterp/Xinterp[0],label='[H2G]')    
    ax3.legend(loc='best',frameon=True,framealpha=0.5)
    ax3.set_xlabel('Guest equiv ([G]0 / [H]0)')
    ax3.set_ylabel('Molfraction')
    plt.tight_layout()
    plt.show()
    return result


#Global fit for 1:2 H:G complex data
def Global12(data):

    def G12(G0,H0,K1,K2):
        G0,H0,K1,K2 = np.asarray(G0), np.asarray(H0), np.asarray(K1), np.asarray(K2)
        try:
            length = len(G0)
            coeffs = np.zeros((length,4))
            for i in range(length):
                coeffs[i] = [K1*K2, K1*(2*K2*H0[i] - K2*G0[i] + 1), K1*(H0[i]-G0[i])+1, -G0[i]]
            roots = np.zeros((len(G0),3))
            R = np.zeros(len(G0))
            for i in range(len(G0)):
                roots[i] = np.roots(coeffs[i])
                R[i] = min(roots[i][roots[i] >= 0])
            return R
        except TypeError:
            coeffs = [K1*K2, K1*(2*K2*H0 - K2*G0 + 1), K1*(H0-G0)+1, -G0]
            R =  np.roots(coeffs)
        return min(R[R >= 0])

    def nmr12(X,K1,K2,delHG,delHG2):
        H0, G0 = X
        G = G12(G0,H0,K1,K2)
        delDel = (delHG*K1*G + delHG2*K1*K2*G**2)/(1+K1*G+K1*K2*G**2)
        return delDel

    def nmr12_dataset(X,params,i):
        K1 = params['K1_%i' % (i+1)].value
        K2 = params['K2_%i' % (i+1)].value
        delHG = params['delDHG1_%i' % (i+1)].value
        delHG2 = params['delDHG2_%i' % (i+1)].value
        return nmr12(X,K1,K2,delHG,delHG2)

    def objective(params, X, data):
        """ calculate total residual for fits to several data sets"""
        dataT = np.array(data.T[2:])
        ndata, nx = dataT.shape
        resid = 0.0*dataT[:]
        # make residual per data set
        for i in range(ndata):
            dataT[i] = dataT[i] - dataT[i][0]
            resid[i, :] = dataT[i, :] - nmr12_dataset(X,params,i)
        # now flatten this to a 1D array, as minimize() needs
        return resid.flatten()

    X = data.iloc[:,0],data.iloc[:,1] 

    dataT = np.array(data.T[2:])
    fit_params = Parameters()
    for iy, y in enumerate(dataT):
        fit_params.add('K1_%i' % (iy+1), value=1000, min=0.0,  max=1e6)
        fit_params.add('K2_%i' % (iy+1), value=100, min=0.0,  max=1e4)
        fit_params.add('delDHG1_%i' % (iy+1), value=1.0, min=-4.0,  max=4.0)
        fit_params.add('delDHG2_%i' % (iy+1), value=-1.0, min=-2.0,  max=2.0)

    for iy in range(2,len(dataT)+1):
        fit_params['K1_%i' % iy].expr='K1_1'
        fit_params['K2_%i' % iy].expr='K2_1'
    
    
    result = minimize(objective, fit_params, args=(X, data))
    #report_fit(result.params)
    
    palette = itertools.cycle(sns.color_palette())
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True)
    plt.rcParams["lines.markeredgewidth"] = 1.5
    interp = np.linspace(min(X[1]),max(X[1]),1000)
    Xinterp = (interp*0.0+X[0][0]), interp
    
    Ginterp = G12(Xinterp[1],Xinterp[0],result.params['K1_1'].value,result.params['K2_1'].value)
    Hinterp = Xinterp[0]/(1+result.params['K1_1'].value*Ginterp + result.params['K1_1'].value*result.params['K2_1'].value * Ginterp * Ginterp)
    HGinterp = result.params['K1_1'].value*Ginterp * Hinterp
    HG2interp = result.params['K2_1'].value * HGinterp * Ginterp
    
    for i in range(len(dataT)):
        c = next(palette)
        y_fit = nmr12_dataset(X, result.params, i)
        Yinterp = nmr12_dataset(Xinterp, result.params, i)
        ax1.plot(X[1]/X[0], dataT[i, :]-dataT[i,0],'o',color=c,markerfacecolor='None')
        ax1.plot(Xinterp[1]/Xinterp[0], Yinterp, '-', color=c)
        ax2.plot(X[1]/X[0],dataT[i,:]-dataT[i,0]-y_fit, 'o--',color=c,markerfacecolor='None')
        ax1.set_ylabel('Δδ')
        ax2.set_ylabel('Residual')
        
        ax2.axhline(y=0,color='k')
    ax3.plot(Xinterp[1]/Xinterp[0], Hinterp/Xinterp[0],label='[H]')
    ax3.plot(Xinterp[1]/Xinterp[0], HGinterp/Xinterp[0],label='[HG]')
    ax3.plot(Xinterp[1]/Xinterp[0], HG2interp/Xinterp[0],label='[HG2]')    
    ax3.legend(loc='best',frameon=True,framealpha=0.5)
    ax3.set_xlabel('Guest equiv ([G]0 / [H]0)')
    ax3.set_ylabel('Molfraction')
    plt.tight_layout()
    plt.show()
    return result

#GLOBAL ANALYSIS 1:1 complex
def Global11(data):
    def HG11(G0,H0,K):
            return 0.5*(G0 + H0 + 1/K - np.sqrt((G0-H0-1/K)**2 + 4*G0/K))
    def nmr11(X,K,delDHG):
        H0, G0 = X
        HG = HG11(G0,H0,K)
        out = delDHG*HG/H0
        return out
    def nmr11_dataset(X,params,i):
        K = params['K_%i' % (i+1)].value
        delDHG = params['delDHG_%i' % (i+1)].value
        return nmr11(X,K,delDHG)

    def objective(params, X, data):
        """ calculate total residual for fits to several data sets"""
        dataT = np.array(data.T[2:])
        ndata, nx = dataT.shape
        resid = 0.0*dataT[:]
        # make residual per data set
        for i in range(ndata):
            dataT[i] = dataT[i] - dataT[i][0]
            resid[i, :] = dataT[i, :] - nmr11_dataset(X,params,i)
        # now flatten this to a 1D array, as minimize() needs
        return resid.flatten()

    X = data.iloc[:,0],data.iloc[:,1] 

    dataT = np.array(data.T[2:])
    fit_params = Parameters()
    for iy, y in enumerate(dataT):
        fit_params.add('K_%i' % (iy+1), value=0.5, min=0.0,  max=1e6)
        fit_params.add('delDHG_%i' % (iy+1), value=0.4, min=-4.0,  max=4.0)
    for iy in range(2,len(dataT)+1):
        fit_params['K_%i' % iy].expr='K_1'


    result = minimize(objective, fit_params, args=(X, data))
    #report_fit(result.params)
    
   
    import matplotlib.cm as cm
    import seaborn as sns
    palette = itertools.cycle(sns.color_palette())
    plt.rcParams["lines.markeredgewidth"] = 1.5
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True)
    
    interp = np.linspace(min(X[1]),max(X[1]),1000)
    Xinterp = (interp*0.0+X[0][0]), interp
              
    HGinterp = HG11(Xinterp[1],Xinterp[0],result.params['K_1'].value)
    Hinterp = Xinterp[0] - HGinterp
              
    for i in range(len(dataT)):
        c = next(palette)
        y_fit = nmr11_dataset(X, result.params, i)
        Yinterp = nmr11_dataset(Xinterp, result.params, i)
        
        ax1.plot(X[1]/X[0], dataT[i, :]-dataT[i,0],'o', color=c,markerfacecolor='None')
        ax1.plot(Xinterp[1]/Xinterp[0], Yinterp, '-', color=c)
        ax2.plot(X[1]/X[0],dataT[i,:]-dataT[i,0]-y_fit, 'o--',color=c,markerfacecolor='None')
    ax3.plot(Xinterp[1]/Xinterp[0],Hinterp/Xinterp[0],label = '[H]')
    ax3.plot(Xinterp[1]/Xinterp[0],HGinterp/Xinterp[0],label = '[HG]')
    ax3.legend(loc='best',frameon=True,framealpha=0.5)
    ax1.set_ylabel('Δδ')
    ax2.set_ylabel('Residual')
    ax3.set_xlabel('Guest equiv ([G]0 / [H]0)')
    ax2.axhline(y=0,color='k')
    ax3.set_ylabel('Molfraction')
    plt.tight_layout()
    plt.show()
    return result
    