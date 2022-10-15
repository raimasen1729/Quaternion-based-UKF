# Quaternion-based-UKF


Built a quaternion based UKF for object pose tracking given IMU readings to avoid orientation singularity 
`estimate_rot.py` tracks the orientation using the IMU data and UKF.

## Vicon vs. UKF
<img src="https://user-images.githubusercontent.com/46754269/196006372-ebae9c3a-e5fa-488a-b329-f26ed9909f5d.png" width="300" height="300"> 

## Gyro vs. UKF
<img src="https://user-images.githubusercontent.com/46754269/196006392-1ecc844d-b8a3-459c-9a0a-2df8f935ad19.png" width="300" height="300"> 

## Reference
> E. Kraft, "A quaternion-based unscented Kalman filter for orientation tracking," Sixth International Conference of Information Fusion, 2003. Proceedings of the, Cairns, Queensland, Australia, 2003, pp. 47-54, doi: 10.1109/ICIF.2003.177425.
