from scipy.integrate import solve_ivp
import numpy as np
import open3d as o3d
from alive_progress import alive_bar

class pointsGen:
    def __init__(self, a, c = np.ones(3)) -> None:
        self.a = a
        self.c = c

    def mu_de(self, t, mu):
        c1, c2, c3 = self.c
        mu_dot = np.zeros(6)
        mu_dot[0] = mu[2]*mu[1]*(1/c3 - 1/c2)
        mu_dot[1] = mu[5] + mu[0]*mu[2]*(1/c1 - 1/c3)
        mu_dot[2] = -mu[4] + mu[0]*mu[1]*(1/c2 - 1/c1)
        mu_dot[3] = mu[2]*mu[4]/c3 - mu[1]*mu[5]/c2
        mu_dot[4] = mu[0]*mu[5]/c1 - mu[2]*mu[3]/c3
        mu_dot[5] = mu[1]*mu[3]/c2 - mu[0]*mu[4]/c1
        return mu_dot
    
    def solve_mu(self):
        self.mu_sol = solve_ivp(self.mu_de, [0, 1], self.a, dense_output=True)
        # t = np.linspace(0,1,100)
        # print(self.mu_sol.sol(t))
        # print(self.mu_sol.status)
    
    def q_de(self, t, q):
        # q: numpy array (16) - can be reshaped to (4,4)
        c1, c2, c3 = self.c
        mu = self.mu_sol.sol(t)
        r_mat = np.zeros((4,4))
        r_mat[0,1] = -mu[2]/c3
        r_mat[0,2] = mu[1]/c2
        r_mat[0,3] = 1.0
        r_mat[1,0] = mu[2]/c3
        r_mat[1,2] = -mu[0]/c1
        r_mat[2,0] = -mu[1]/c2
        r_mat[2,1] = mu[0]/c1
        q_dot = np.matmul(q.reshape((4,4)), r_mat)
        return q_dot.reshape((16))
    
    def solve_q(self):
        self.q_sol = solve_ivp(self.q_de, [0, 1], np.identity(4).reshape((16)), dense_output=True)
        # t = np.linspace(0,1,100)
        # print(self.q_sol.sol(t).shape)

    def mj_de(self, t, mj):
        # q: numpy array (6*6*2=72) - can be reshaped to (12,6): M: first 6 rows; J: last 6 rows
        c1, c2, c3 = self.c
        mu = self.mu_sol.sol(t)
        MJ = mj.reshape((12, 6))
        M = MJ[0:6]
        J = MJ[6:12]
        F = np.array([
            [0,                     mu[2]*(1/c3 - 1/c2),    mu[1]*(1/c3 - 1/c2),    0,          0,          0           ],
            [mu[2]*(1/c1 - 1/c3),   0,                      mu[0]*(1/c1 - 1/c3),    0,          0,          1           ],
            [mu[1]*(1/c2 - 1/c1),   mu[0]*(1/c2 - 1/c1),    0,                      0,          -1,         0           ],
            [0,                     -mu[5]/c2,              mu[4]/c3,               0,          mu[2]/c3,   -mu[1]/c2   ],
            [mu[5]/c1,              0,                      -mu[3]/c3,              -mu[2]/c3,  0,          mu[0]/c1    ],
            [-mu[4]/c1,             mu[3]/c2,               0,                      mu[1]/c2,   -mu[0]/c1,  0           ]
        ])

        G = np.zeros((6,6))
        G[0,0] = 1/c1
        G[1,1] = 1/c2
        G[2,2] = 1/c3

        H = np.array([
            [0,         mu[2]/c3,   -mu[1]/c2,  0,          0,          0           ],
            [-mu[2]/c3, 0,          mu[0]/c1,   0,          0,          0           ],
            [mu[1]/c2,  -mu[0]/c1,  0,          0,          0,          0           ],
            [0,         0,          0,          0,          mu[2]/c3,   -mu[1]/c2   ],
            [0,         0,          1,          -mu[2]/c3,  0,          mu[0]/c1,    ],
            [0,         -1,         0,          mu[1]/c2,   -mu[0]/c1,  0,           ]
        ])

        MJ_dot = np.zeros((12, 6))
        MJ_dot[0:6] = np.matmul(F,M)
        MJ_dot[6:12] = np.matmul(G,M) + np.matmul(H,J)
        return MJ_dot.reshape((72))
    
    def zero_det_j(self, t, mj):
        if t > 0.001:
            MJ = mj.reshape((12, 6))
            J = MJ[6:12]
            return np.linalg.det(J)
        else:
            return 1.0
    zero_det_j.terminal = True

    def solve_mj(self):
        self.mj_sol = solve_ivp(self.mj_de, [0, 1], np.vstack((np.identity(6), np.zeros((6,6)))).reshape((72)), dense_output=True, events=self.zero_det_j)
        # print(self.mj_sol.status)

    def vis(self):
        t = np.linspace(0,1,100)
        xyz = self.q_sol.sol(t).reshape((4,4,-1))[0:3, 3, :].T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        viewer.add_geometry(pcd)
        viewer.add_geometry(coor)
        viewer.run()
        viewer.destroy_window()

def vis(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pcd)
    viewer.add_geometry(coor)
    viewer.run()
    viewer.destroy_window()

num_case = 0
num_pts = 50
gen_cases = np.zeros((10000, num_pts, 16))
gen_a = np.zeros((10000, 6))
with alive_bar(10000) as bar:
    while num_case < 10000:
        a = np.concatenate((np.random.rand(3)*2*np.pi-np.pi, np.random.rand(3)*100-50))
        points_gen = pointsGen(a=a)
        points_gen.solve_mu()
        points_gen.solve_q()
        points_gen.solve_mj()
        if points_gen.mj_sol.status == 0:
            t = np.linspace(0,1,num_pts)
            # gen_cases[num_case] = points_gen.q_sol.sol(t).reshape((4,4,-1))[0:3, 3, :].T
            gen_cases[num_case] = points_gen.q_sol.sol(t).T
            gen_a[num_case] = a
            num_case += 1
            bar()
        #     points_gen.vis()
        # else:
        #     print("not optimal")
np.save("points_50_se3.npy", gen_cases)
np.save("a_50_se3.npy", gen_a)