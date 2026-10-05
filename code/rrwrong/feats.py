"""Catalogue + imaging features for P(Redrock wrong). No truth columns in here."""
import numpy as np, pandas as pd
MORPH = {"PSF":0,"REX":1,"EXP":2,"DEV":3,"SER":4,"DUP":5}
SPEC = {"GALAXY":0,"QSO":1,"STAR":2}
def mag(f):
    return 22.5 - 2.5*np.log10(np.clip(f, 1e-3, None))
def features(d, tgt_col):
    f = pd.DataFrame(index=d.index)
    z = d.Z.to_numpy()
    f["z"] = z
    f["logdchi2"] = np.log10(np.clip(d.DELTACHI2, 1e-2, None))
    f["zerr"] = np.log10(np.clip(d.ZERR/(1+z), 1e-7, None))
    f["rchi2"] = d.CHI2/np.clip(d.NPIXELS, 1, None)
    f["npix"] = d.NPIXELS
    f["zwarn"] = d.ZWARN
    f["spectype"] = d.SPECTYPE.str.strip().map(SPEC).fillna(3)
    c = np.stack([d[f"COEFF_{i}"] for i in range(10)], 1)
    n = np.abs(c).sum(1, keepdims=True) + 1e-30
    for i in range(5): f[f"coef{i}"] = c[:, i]/n[:, 0]
    f["coefmax"] = np.log10(np.abs(c).max(1)+1e-30)
    for t in ["ELG","LRG","QSO","BGS","LYA"]:
        f["tsnr_"+t] = np.log10(np.clip(d["TSNR2_"+t], 1e-3, None))
    f["exptime"] = d.COADD_EXPTIME
    m = {b: mag(d["FLUX_"+b]) for b in ["G","R","Z","W1","W2"]}
    for b in m: f["mag_"+b] = m[b]
    f["g_r"] = m["G"]-m["R"]; f["r_z"] = m["R"]-m["Z"]; f["z_w1"] = m["Z"]-m["W1"]; f["w1_w2"] = m["W1"]-m["W2"]
    f["fibmag_r"] = mag(d.FIBERFLUX_R); f["fibmag_z"] = mag(d.FIBERFLUX_Z)
    f["fibfrac_r"] = d.FIBERFLUX_R/np.clip(d.FLUX_R, 1e-3, None)
    for b in ["G","R","Z","W1"]:
        f["sn_"+b] = d["FLUX_"+b]*np.sqrt(np.clip(d["FLUX_IVAR_"+b], 0, None))
    f["ebv"] = d.EBV
    f["morph"] = d.MORPHTYPE.astype(str).str.strip().map(MORPH).fillna(6)
    f["shape_r"] = d.SHAPE_R; f["sersic"] = d.SERSIC
    g = d.GAIA_PHOT_G_MEAN_MAG.where(d.GAIA_PHOT_G_MEAN_MAG > 0)
    f["gaia_g"] = g; f["gaia_bprp"] = (d.GAIA_PHOT_BP_MEAN_MAG-d.GAIA_PHOT_RP_MEAN_MAG).where(g.notna())
    f["parallax"] = d.PARALLAX.where(g.notna()); f["pm"] = np.hypot(d.PMRA, d.PMDEC).where(g.notna())
    f["maskbits"] = d.MASKBITS
    t = d[tgt_col].to_numpy().astype(np.int64)
    for name, b in [("t_lrg",0),("t_elg",1),("t_qso",2),("t_bgs",60),("t_mws",61)]:
        f[name] = (t >> b) & 1
    return f.astype("f4")
CAT = ["spectype","morph"]
