cd VersionControl/nucleosynthesis_paper4_code_MAGFIELD/vis/python
ano="7_INNER_NB4"
base="/Volumes/RESEARCHUSB/Research"
outp="${base}/Animations/${ano}/"
athdfdir="${base}/Outputs/MagField/mag_field/nucleosynthesis_paper4_code/out_rho12_QW_B3e15_P20ms_L8e51_vrvth0_512_256_allheatingterms_CORRECTEDperfectBCs/"
rmin=0.0
rmax=0.005
phimin=-0.008
phimax=0.008
dens=20
boundary=20000000.0
python3 animate.py -p ${athdfdir} -o ${outp} -oi "${base}/animation${ano}.gif" --create -d ${dens} -vrmin ${rmin} -vrmax ${rmax} -vphimin ${phimin} -vphimax ${phimax} -ob ${boundary}