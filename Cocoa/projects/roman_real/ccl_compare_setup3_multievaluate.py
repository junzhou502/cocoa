import numpy as np
import pytest
import pyccl as ccl
from pyccl import CCLWarning

from pyccl.pk2d import Pk2D
from pyccl.correlations import correlation

import os
import time



rootdir = '/project/chihway/junzhou/Code-comparison'
workfolder = './cocoa/Cocoa/projects/roman_real/chains/roman_setup3_multievaluate/'
outname = 'chis.txt'

filename = '1.txt'
filepath = os.path.join(rootdir, workfolder)
cosmologys = np.loadtxt(filepath+filename)

with open(os.path.join(rootdir, workfolder, outname), 'a', encoding='utf-8') as f:
    f.write("#As_1e9    ns    H0    omegab    omgeam    omegach2    w0pwa    w    xi+    xi+_msk    xi-    xi-_msk    gammat    gammat_msk    wtheta    wtheta_msk    msk\n")

As_1e9, ns, H0, omegab, omegam, omegach2, w0pwa, w = [],[],[],[],[],[],[],[]
idx = [2, 3, 4, 5, 6, 51, 8, 9]
for i in range(len(cosmologys)):
    As_1e9.append(cosmologys[i, idx[0]])
    ns.append(cosmologys[i, idx[1]])
    H0.append(cosmologys[i, idx[2]])
    omegab.append(cosmologys[i, idx[3]])
    omegam.append(cosmologys[i, idx[4]])
    omegach2.append(cosmologys[i, idx[5]])
    w0pwa.append(cosmologys[i, idx[6]])
    w.append(cosmologys[i, idx[7]])

#load covariance matrix and dv from cosmolike
cov_file = './cocoa/Cocoa/projects/roman_real/data/cov_roman'
mask_file = './cocoa/Cocoa/projects/roman_real/data/roman_Y3.mask'
mask = np.loadtxt(os.path.join(rootdir,mask_file))[:,1].astype(bool)
cov_raw = np.loadtxt(os.path.join(rootdir,cov_file))
ncov = int(np.max(cov_raw[:,0]))+1
cov = np.zeros((ncov, ncov))
for i in range(len(cov_raw)):
    ii = int(cov_raw[i, 0])
    jj = int(cov_raw[i, 1])
    element = cov_raw[i,8] + cov_raw[i,9]
    cov[ii,jj] = element
    cov[jj,ii] = element
nsrcs = 8
nlens = 10
ntheta = 20
starts = [0, int(nsrcs*(nsrcs+1)/2*ntheta), int(nsrcs*(nsrcs+1)*ntheta), int((nsrcs*(nsrcs+1) + nlens*nsrcs)*ntheta), int((nsrcs*(nsrcs+1) + nlens*nsrcs + nlens)*ntheta)]
invcov_seg = []
invcov_seg_masked = []
invcov_masked = np.linalg.pinv(cov[mask, :][:, mask])
for i in range(4):
    l = starts[i]
    r = starts[i+1]
    mask_seg = mask[l:r]
    cov_seg = cov[l:r, :][:, l:r]
    cov_seg_masked = cov_seg[mask_seg,:][:,mask_seg]
    invcov_seg.append(np.linalg.pinv(cov_seg))
    invcov_seg_masked.append(np.linalg.pinv(cov_seg_masked))

t0 = time.time()
for i_cosmo in range(len(cosmologys)):
    
    h = H0[i_cosmo]/100.0
    
    COSMO = ccl.Cosmology(
    Omega_c=omegach2[i_cosmo]/h**2,
    Omega_b=omegab[i_cosmo],
    Omega_k=0,
    h=h,
    w0=w[i_cosmo],
    wa=w0pwa[i_cosmo] - w[i_cosmo],
    A_s=As_1e9[i_cosmo]/1e9,
    n_s=ns[i_cosmo],
    m_nu=0.06,
    Neff=3.046,
    mass_split='single',
    transfer_function='boltzmann_camb',
    matter_power_spectrum='camb',
    extra_parameters = {"camb": {"halofit_version": "takahashi",
                                 'AccuracyBoost': 1.0,
                                 'kmax':15,
                                 'dark_energy_model': 'ppf',
                                 'accurate_massive_neutrino_transfer': False,
                                 'k_per_logint': 15,
                                 }}
    )

    #-----------------------------------------------------------------#
    #---------------based on discussion in section 0------------------#
    #------------------we use this power spectrum---------------------#
    #-----------------------------------------------------------------# 
    
    z_pk_cocoa = np.loadtxt(filepath+f'z_pk_{i_cosmo+1}.txt')
    k_pk_cocoa = np.loadtxt(filepath+f'k_pk_{i_cosmo+1}.txt')   #1/Mpc
    lnPk_cocoa = np.loadtxt(filepath+f'pknl_{i_cosmo+1}.txt').reshape(len(k_pk_cocoa),len(z_pk_cocoa) ).T   #Mpc/h^3
    lnPk_lin_cocoa = np.loadtxt(filepath+f'pkln_{i_cosmo+1}.txt').reshape(len(k_pk_cocoa),len(z_pk_cocoa) ).T   #Mpc/h^3
    lnPk_cocoa = lnPk_cocoa[::-1,:] - 3*np.log(h)
    lnPk_lin_cocoa = lnPk_lin_cocoa[::-1,:] - 3*np.log(h)
    a_pk_cocoa = 1/(1+z_pk_cocoa)[::-1]
    lnk_pk_cocoa = np.log(k_pk_cocoa)
    pk2_cocoa = Pk2D(a_arr=a_pk_cocoa, lk_arr=lnk_pk_cocoa, pk_arr=lnPk_cocoa, is_logp=True)
    pk2_lin_cocoa = Pk2D(a_arr=a_pk_cocoa, lk_arr=lnk_pk_cocoa, pk_arr=lnPk_lin_cocoa, is_logp=True)

    #---------------------------------------------------------------------------------------------#
    #-------------------this Pk is further injected into power specetrum calculation--------------#
    #---------------------------------------------------------------------------------------------#
    #set up lens source sample
    srcs_nzs = np.loadtxt(os.path.join(rootdir,'./cocoa/Cocoa/projects/roman_real/data/fiducial.nz'))
    lens_nzs = np.loadtxt(os.path.join(rootdir,'./cocoa/Cocoa/projects/roman_real/data/lsst_lens_4cosmolike.nz'))
    
    z_srcs = srcs_nzs[:,0]+0.005
    srcs_nz = srcs_nzs[:,1:]
    z_lens = lens_nzs[:,0]+0.005
    lens_nz = lens_nzs[:,1:]
    
    srcs = []
    lens = []
    nsrcs = 8
    nlens = 10
    gbias =  [1.09,1.15,1.21,1.27,1.33,1.40,1.46,1.53,1.60,1.67]
    ggl_exclude = []
    
    for i in range(nsrcs):
        srcs.append(ccl.WeakLensingTracer(COSMO, dndz=(z_srcs, srcs_nz[:,i]), has_shear=True, n_samples=400))
    for i in range(nlens):
        lens.append(ccl.NumberCountsTracer(COSMO, dndz=(z_lens, lens_nz[:,i]), bias=(z_lens,np.ones_like(z_lens)*gbias[i]), has_rsd=True, n_samples=400))
    
    
    #calculate the power spectrum for calculation of correlation function
    logLMIN = np.log(20)
    logLMAX = np.log(60000+1)
    NCell = 300
    dlogL = (logLMAX - logLMIN)/(NCell - 1)
    
    ells = np.zeros(20+NCell)
    for i in range(20):
        ells[i] = i
    for i in range(NCell):
        ells[i+20] = np.exp(logLMIN + dlogL*i)
    ells = ells[1:]
    
    corrs = []
    for i in range(nsrcs):
        for j in range(i,nsrcs):
            corrs.append(ccl.angular_cl(COSMO, srcs[i], srcs[j], ells, p_of_k_a=pk2_cocoa, l_limber=-1) )
    
    for i in range(nlens):
        for j in range(nsrcs):
            if [i,j] in ggl_exclude:
                continue
            corrs.append(ccl.angular_cl(COSMO, lens[i], srcs[j], ells, p_of_k_a=pk2_cocoa, l_limber=-1) )
            
    for i in range(nlens):
        corrs.append(ccl.angular_cl(COSMO, lens[i], lens[i], ells, p_of_k_a=pk2_cocoa,l_limber=150,
                                        limber_max_error=0.000001,
                                        non_limber_integration_method='FKEM',
                                        fkem_chi_min = None,
                                        fkem_Nchi = 500,
                                        p_of_k_a_lin = pk2_lin_cocoa,))    
    #calculate the correlation function
    tmin = 1.
    tmax = 500.
    ntheta = 20
    
    logtmin = np.log(tmin)
    logtmax = np.log(tmax)
    logdt=(logtmax - logtmin)/ntheta
    fac = (2./3.)
    thetas = np.zeros(ntheta)
    
    for i in range(ntheta):
        thetamin = np.exp(logtmin + (i + 0.)*logdt)
        thetamax = np.exp(logtmin + (i + 1.)*logdt)
        thetas[i] = fac * (thetamax**3 - thetamin**3) / (thetamax*thetamax    - thetamin*thetamin)
    thetas /= 60
    
    ncombo1 = int(nsrcs*(nsrcs+1)/2)
    ncombo2 = int(nsrcs*nlens)
    ncombo3 = int(nlens)
    
    xip, xim, gammat, wtheta = [],[],[],[]
    for i in range(ncombo1):
        xip.append(correlation(COSMO, ell=ells, C_ell=corrs[i], theta=thetas, type='GG+', method='FFTLog'))
        xim.append(correlation(COSMO, ell=ells, C_ell=corrs[i], theta=thetas, type='GG-', method='FFTLog'))
    xip = np.concatenate(xip)
    xim = np.concatenate(xim)
    
    for i in range(ncombo1, ncombo1+ncombo2):
        gammat.append(correlation(COSMO, ell=ells, C_ell=corrs[i], theta=thetas, type='NG', method='Legendre'))
    gammat = np.concatenate(gammat)
    
    for i in range(ncombo1 + ncombo2, ncombo1 + ncombo2 + ncombo3):
        wtheta.append(correlation(COSMO, ell=ells, C_ell=corrs[i], theta=thetas, type='NN', method='Legendre'))
    wtheta = np.concatenate(wtheta)
     
    dv_ccl = np.concatenate((xip,xim,gammat,wtheta))

    dv_cosmolike = np.loadtxt(os.path.join(rootdir,f'./cocoa/Cocoa/projects/roman_real/chains/roman_setup3_multievaluate/roman.modelvector_{i_cosmo+1}'))[:,1]

    starts = [0, int(nsrcs*(nsrcs+1)/2*ntheta), int(nsrcs*(nsrcs+1)*ntheta), int((nsrcs*(nsrcs+1) + nlens*nsrcs)*ntheta), int((nsrcs*(nsrcs+1) + nlens*nsrcs + nlens)*ntheta)]
    output = [As_1e9[i_cosmo], ns[i_cosmo], H0[i_cosmo], omegab[i_cosmo], omegach2[i_cosmo], w[i_cosmo]]
    for i in range(4):
        l = starts[i]
        r = starts[i+1]
        
        mask_seg = mask[l:r]
        dv_ccl_seg = dv_ccl[l:r]
        dv_cosmolike_seg = dv_cosmolike[l:r]
        cov_seg = cov[l:r, :][:, l:r]
        delta_seg = dv_ccl_seg - dv_cosmolike_seg
        chi2_seg = delta_seg@invcov_seg[i]@delta_seg
        output.append(chi2_seg)
        dv_ccl_seg_masked = dv_ccl_seg[mask_seg]
        dv_cosmolike_seg_masked = dv_cosmolike_seg[mask_seg]
        cov_seg_masked = cov_seg[mask_seg,:][:,mask_seg]
        delta_seg_masked = dv_ccl_seg_masked - dv_cosmolike_seg_masked
        chi2_seg_masked = delta_seg_masked@invcov_seg_masked[i]@delta_seg_masked
        output.append(chi2_seg_masked)
    delta_masked = dv_ccl[mask]-dv_cosmolike[mask]
    output.append(delta_masked@invcov_masked@delta_masked)

    with open(os.path.join(rootdir, workfolder, outname), 'a', encoding='utf-8') as f:
        row = " ".join(map(str, output))
        f.write(row + "\n")
    t1 = time.time()
    print(f'{i_cosmo+1} finished, time lapse {t1-t0:.2f}', flush=True)