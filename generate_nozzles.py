#!/usr/bin/env python3
"""
RoachHunter 四合一旋轉殺蟑噴頭產生器
==================================
結構（類似園藝水管的多段噴頭）：

  turret_base  按壓座：壓在噴霧罐閥桿上，前方有出液口（O 型環密封）與轉軸孔
  turret_disc  旋轉盤：盤上 4 個噴嘴，每 90° 一個，轉到「轉軸正下方」的那個就是使用中的噴嘴
                 ① 遠距：收斂式 0.6 mm 噴孔，柱狀直射
                 ② 近距：1.2 mm 喉部 + 擴散錐，圓錐霧狀
                 ③ 大面積：7 孔扇形，水平 ±45°
                 ④ 縫隙：55 mm 細長管
  stem_fit_gauge 量測規：找出最合適的閥桿套孔尺寸

組裝五金：M3×20 螺絲 1、M3 尼龍防鬆螺帽 1、M3 墊片 1、O 型環（內徑 3 × 線徑 1.5 mm）1

用法：
  python3 generate_nozzles.py                 # 產生旋轉盤 + 3.0~4.2 mm 全部按壓座
  python3 generate_nozzles.py --bore 3.5      # 只產生指定套孔尺寸的按壓座
需求：pip install manifold3d trimesh
"""
import argparse
import os

import numpy as np
import manifold3d as mf
import trimesh

SEG = 96
EPS = 0.02

# ---------------------------------------------------------------- 尺寸參數

BODY_R = 9.0           # 按壓座半徑
BODY_H = 14.0
SOCKET_DEPTH = 4.5     # 閥桿套孔深度
CH_Z = 8.0             # 出液口高度
NOZ_R = 12.0           # 噴嘴所在圓半徑（= 轉軸到出液口距離）
AXIS_Z = CH_Z + NOZ_R  # 轉軸高度
X_PLATE0 = 5.0         # 前板背面
X_FACE = 15.0          # 前板正面（密封面）
DISC_R = 19.0          # 旋轉盤半徑
DISC_T = 6.0           # 旋轉盤厚度
INLET_R = 1.0          # 盤上入口孔半徑
PORT_R = 0.8           # 出液口半徑
ORING_ID, ORING_CS = 3.0, 1.5
DETENT_R = 6.0         # 定位卡點所在半徑
M3_CLEAR = 1.7

# ---------------------------------------------------------------- 工具


def cyl_z(x, y, z0, h, r, r2=None, seg=SEG):
    return mf.Manifold.cylinder(h, r, r if r2 is None else r2, seg).translate([x, y, z0])


def cyl_x(x0, length, y, z, r, r2=None, seg=SEG):
    return mf.Manifold.cylinder(length, r, r if r2 is None else r2, seg).rotate([0, 90, 0]).translate([x0, y, z])


def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def hull(*parts):
    return mf.Manifold.batch_hull(list(parts))


def sphere(p, r, seg=48):
    return mf.Manifold.sphere(r, seg).translate(list(p))


def capsule(p1, p2, r):
    return hull(sphere(p1, r, 32), sphere(p2, r, 32))


# ================================================================ 按壓座（世界座標，直接照此方向列印）


def turret_base(bore):
    body = cyl_z(0, 0, 0, BODY_H, BODY_R)
    plate = hull(
        cyl_x(X_PLATE0, X_FACE - X_PLATE0, 0, CH_Z, 7.0),
        cyl_x(X_PLATE0, X_FACE - X_PLATE0, 0, AXIS_Z, 6.0),
        box(X_PLATE0, X_FACE, -7.0, 7.0, 0, 1.0),
    )
    solid = body + plate
    solid -= sphere((0, 0, BODY_H + 30 - 0.8), 30, 128) ^ cyl_z(0, 0, BODY_H - 2, 3, BODY_R - 1.0)  # 頂面指壓淺碟

    r = bore / 2
    cav = cyl_z(0, 0, -EPS, SOCKET_DEPTH + EPS, r)                 # 閥桿套孔
    cav += cyl_z(0, 0, -EPS, 0.8, r + 0.5, r)                       # 入口導角
    cav += cyl_z(0, 0, SOCKET_DEPTH - EPS, CH_Z - SOCKET_DEPTH, 0.8)
    cav += capsule((0, 0, CH_Z), (X_FACE - 2.0, 0, CH_Z), 1.0)
    cav += cyl_x(X_FACE - 2.5, 3.0, 0, CH_Z, PORT_R)                # 出液口

    # O 型環端面溝槽（深度 = 線徑 72%，留約 0.4 mm 壓縮量）
    g_in, g_out, g_d = ORING_ID / 2, ORING_ID / 2 + ORING_CS + 0.2, ORING_CS * 0.72
    cav += cyl_x(X_FACE - g_d, g_d + 1, 0, CH_Z, g_out) - cyl_x(X_FACE - g_d - 1, g_d + 3, 0, CH_Z, g_in)

    # 轉軸螺絲孔，背面沉頭（螺絲由後往前穿，螺帽在旋轉盤正面）
    cav += cyl_x(X_PLATE0 - 1, X_FACE - X_PLATE0 + 2, 0, AXIS_Z, M3_CLEAR)
    cav += cyl_x(X_PLATE0 - 1, 4.0, 0, AXIS_Z, 3.0)
    solid -= cav

    # 定位凸點：卡進旋轉盤背面凹坑，轉到位會「喀」一聲
    bz = AXIS_Z - DETENT_R
    solid += sphere((X_FACE - 0.6, 0, bz), 1.0, 32) - box(X_FACE - 3, X_FACE - 0.6 + EPS, -2, 2, bz - 2, bz + 2)
    return solid


# ================================================================ 旋轉盤
# 本地座標：Z = 轉軸 = 噴射方向，背面 z=0 貼列印平台，噴嘴朝上（細孔垂直列印最圓最準）。
# 每個噴嘴建在 (+NOZ_R, 0) 再繞 Z 轉到各自角度。
# 組裝 rotate([0,90,0]) 後本地 +X → 世界 -Z，所以 (+NOZ_R, 0) 位於轉軸正下方 = 出液口。

T = DISC_T


def noz_long(orifice=0.6):
    L = 20.0
    shell = cyl_z(NOZ_R, 0, T - EPS, L, 4.5, 2.6)
    ro = orifice / 2
    cav = cyl_z(NOZ_R, 0, -EPS, T + L - 4.0, INLET_R)
    cav += cyl_z(NOZ_R, 0, T + L - 4.0 - EPS, 2.5, INLET_R, ro)     # 收斂段
    cav += cyl_z(NOZ_R, 0, T + L - 1.6, 2.0, ro)                     # 噴孔直段 L/D≈3
    return shell, cav


def noz_short():
    L = 7.0
    shell = cyl_z(NOZ_R, 0, T - EPS, L, 6.0)
    cav = cyl_z(NOZ_R, 0, -EPS, T + 1.0, INLET_R)
    cav += cyl_z(NOZ_R, 0, T + 1.0 - EPS, 1.0, 0.6)                  # 1.2 mm 喉部
    cav += cyl_z(NOZ_R, 0, T + 2.0 - EPS, L - 2.0 + 0.3, 0.6, 4.5)   # 擴散錐
    return shell, cav


def noz_fan(hole=0.7, n=7, spread=45.0):
    pz = T + 3.0                       # 分配腔中心
    Rh = 8.0                           # 扇頭半徑
    dome = sphere((NOZ_R, 0, pz), Rh, 96) ^ box(NOZ_R - 3.5, NOZ_R + 3.5, -Rh, Rh, T - EPS, pz + Rh)
    shell = dome + cyl_z(NOZ_R, 0, T - EPS, 2.0, 3.5)
    cav = cyl_z(NOZ_R, 0, -EPS, pz, INLET_R)
    cav += sphere((NOZ_R, 0, pz), 2.2)                               # 分配腔
    # 扇形孔沿切線（本地 Y）展開；轉到使用位置時就是水平展開
    c = np.array([NOZ_R, 0, pz])
    for a in np.radians(np.linspace(-spread, spread, n)):
        d = np.array([0.0, np.sin(a), np.cos(a)])
        cav += capsule(c + d * 1.5, c + d * (Rh + 1.0), hole / 2)
    return shell, cav


def noz_crevice(tip_hole=1.2):
    L = 55.0
    shell = cyl_z(NOZ_R, 0, T - EPS, 8.0, 5.0, 2.4) + cyl_z(NOZ_R, 0, T - EPS, L, 2.4)
    cav = cyl_z(NOZ_R, 0, -EPS, T + 2.0, INLET_R)
    cav += cyl_z(NOZ_R, 0, T + 2.0 - EPS, 0.6, INLET_R, tip_hole / 2)
    cav += cyl_z(NOZ_R, 0, T + 2.0, L + 1, tip_hole / 2)
    return shell, cav


NOZZLES = [("遠距", noz_long, 0), ("近距", noz_short, 90), ("大面積", noz_fan, 180), ("縫隙", noz_crevice, 270)]


def turret_disc(orifice=0.6):
    solid = cyl_z(0, 0, 0, T, DISC_R)
    cav = cyl_z(0, 0, -1, T + 2, M3_CLEAR + 0.1)                     # 轉軸孔（可自由轉動）
    for a in np.radians(np.arange(7.5, 360, 15)):                     # 外緣防滑齒
        cav += cyl_z(DISC_R * np.cos(a), DISC_R * np.sin(a), -1, T + 2, 1.0, seg=24)
    for i, (name, fn, ang) in enumerate(NOZZLES):
        shell, c = fn(orifice) if fn is noz_long else fn()
        solid += shell.rotate([0, 0, ang])
        cav += c.rotate([0, 0, ang])
        a = np.radians(ang)
        cav += sphere((DETENT_R * np.cos(a), DETENT_R * np.sin(a), 0), 1.25, 32)   # 背面定位凹坑
        # 正面編號：轉軸旁刻 1~4 個點，朝向對應噴嘴
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        for k in range(i + 1):
            q = rot @ np.array([5.0, (k - i / 2) * 1.6])
            cav += cyl_z(q[0], q[1], T - 0.6, 1.0, 0.5, seg=16)
    return solid - cav


def assembly(bore, orifice=0.6, active=0):
    """組裝示意（僅供預覽）：active = 0~3 代表轉到第幾個噴嘴。"""
    disc = turret_disc(orifice).rotate([0, 0, -NOZZLES[active][2]])
    return turret_base(bore), disc.rotate([0, 90, 0]).translate([X_FACE, 0, AXIS_Z])


# ================================================================ 量測規


def fit_gauge(sizes=(3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2)):
    pitch = 9.0
    plate = box(0, pitch * len(sizes) + 4, 0, 14, 0, SOCKET_DEPTH + 1.5)
    for i, d in enumerate(sizes):
        x = 6.5 + i * pitch
        plate -= cyl_z(x, 6, -EPS, SOCKET_DEPTH + 2, d / 2)
        plate -= cyl_z(x, 6, -EPS, 0.8, d / 2 + 0.5, d / 2)
        for k in range(i + 1):                                         # 刻線數 = 第幾孔
            plate -= box(x - 3 + k, x - 2.5 + k, 11.2, 13.2, SOCKET_DEPTH + 1.0, SOCKET_DEPTH + 2)
    return plate


# ================================================================ 輸出


def clean(man):
    parts = [p for p in man.decompose() if p.volume() > 1.0]          # 去除數值誤差碎片
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add) if len(parts) > 1 else parts[0]


def save(man, path):
    man = clean(man)
    m = man.to_mesh()
    tm = trimesh.Trimesh(vertices=m.vert_properties[:, :3], faces=m.tri_verts, process=False)
    assert man.status() == mf.Error.NoError and tm.is_watertight, f"{path} 不是封閉實體"
    tm.export(path)
    bb = tm.bounds
    print(f"  {path:44s} {bb[1][0]-bb[0][0]:5.1f} x {bb[1][1]-bb[0][1]:5.1f} x "
          f"{bb[1][2]-bb[0][2]:5.1f} mm  體積 {tm.volume/1000:5.2f} cm³  通孔數 {man.genus()}")
    return tm


def main():
    ap = argparse.ArgumentParser(description="RoachHunter 四合一旋轉噴頭 STL 產生器")
    ap.add_argument("--bore", type=float, help="只產生此套孔直徑(mm)的按壓座；不給則產生 3.0~4.2 全部")
    ap.add_argument("--orifice", type=float, default=0.6, help="遠距噴孔直徑 mm（預設 0.6）")
    ap.add_argument("--out", default="stl", help="輸出資料夾")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/turret_base", exist_ok=True)
    save(turret_disc(a.orifice), f"{a.out}/turret_disc.stl")
    save(fit_gauge(), f"{a.out}/stem_fit_gauge.stl")
    for b in ([a.bore] if a.bore else [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2]):
        save(turret_base(b), f"{a.out}/turret_base/turret_base_bore_{b:.1f}mm.stl")


if __name__ == "__main__":
    main()
