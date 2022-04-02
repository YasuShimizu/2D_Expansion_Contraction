import main
import os

#'Q10-1' #Q=10 急拡　実験と同じ初期河床
#'Q10-2' #Q=10 急拡　一定勾配初期河床
#'Q15-1' #Q=15 急縮　実験と同じ初期河床
#'Q15-2' #Q=15 急縮　一定勾配初期河床

runs=["Q10-1","Q10-2","Q15-1","Q15-2"]
#runs=["Q15-1","Q15-2"]
for cn in runs:

    os.system("del /Q "+cn+"\\*.png")
    os.system("del /Q "+cn+"\\*.mp4")    
    os.system("del /Q "+cn+"\\bed\\*.png")    
    os.system("del /Q "+cn+"\\png\\*.png")
    os.system("del /Q "+cn+"\\png_q\\*.png")

    main.nays2d(cn)

    os.system("copy /Y *.png "+cn)
    os.system("copy /Y *.mp4 "+cn)
    cmd="copy /Y png\\*.png "+cn+"\\png"
    os.system(cmd)
    cmd="copy /Y png_q\\*.png "+cn+"\\png_q"
    os.system(cmd)
    cmd="copy /Y bed\\*.png "+cn+"\\bed"
    os.system(cmd)