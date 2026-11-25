#!/usr/bin/env python3
"""
RoachHunter 殺蟑噴頭產生器
=========================
產生 4 款可直接替換「噴霧罐（殺蟲劑氣霧罐）」原廠按壓頭的噴頭 STL，外加一個閥桿尺寸量測規。

  1_long_range.stl   遠距：直線柱狀噴射，收斂式小噴孔
  2_short_range.stl  近距：圓錐霧化，擴散式出口
  3_wide_area.stl    大面積：扇形 7 孔，水平展開 ±45°
  4_crevice.stl      縫隙：75 mm 細長管，伸入牆縫、櫃底、冰箱後方
  stem_fit_gauge.stl 量測規：找出最適合你那罐的套孔尺寸

用法：
  python3 generate_nozzles.py                # 預設套孔 3.4 mm
  python3 generate_nozzles.py --bore 3.6     # 改套孔直徑（依量測規結果）
  python3 generate_nozzles.py --orifice 0.8  # 改遠距噴孔直徑

座標：X = 噴射方向，Z = 列印時向上，底面 z=0 平貼列印板，免支撐。
需求：pip install manifold3d trimesh
"""
import argparse
import os

import numpy as np
import manifold3d as mf
import trimesh

SEG = 96          # 圓的分段數
EPS = 0.02

# ---------------------------------------------------------------- 基本工具


def cyl_z(x, y, z0, h, r, r2=None):
    """沿 Z 軸的圓柱 / 圓錐，底面在 z0。"""
    return mf.Manifold.cylinder(h, r, r if r2 is None else r2, SEG).translate([x, y, z0])


def disc_x(x, y, z, r, t=EPS):
    """垂直於 X 軸的極薄圓盤，用來做 hull 斷面。"""
    return mf.Manifold.cylinder(t, r, r, SEG).rotate([0, 90, 0]).translate([x, y, z])


def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def hull(*parts):
    return mf.Manifold.batch_hull(list(parts))


def capsule(p1, p2, r):
    """兩點之間的圓柱流道（兩端球形）。"""
    s1 = mf.Manifold.sphere(r, 32).translate(list(p1))
    s2 = mf.Manifold.sphere(r, 32).translate(list(p2))
    return hull(s1, s2)


def cone_x(x0, x1, y, z, r0, r1):
    """沿 X 軸的錐形流道。"""
    return hull(disc_x(x0, y, z, r0), disc_x(x1 - EPS, y, z, r1))


def flat_bottom_section(x, z_axis, r, width=None):
    """管身斷面：上半圓 + 平底貼板（免支撐）。"""
    w = 2 * r if width is None else width
    return hull(disc_x(x, 0, z_axis, r), box(x, x + EPS, -w / 2, w / 2, 0, 0.6))


# ---------------------------------------------------------------- 共用按壓座

BODY_R = 9.0      # 按壓座半徑（直徑 18 mm，和多數原廠按壓頭相近）
BODY_H = 14.0
SOCKET_DEPTH = 4.5
CH_Z = 7.0        # 主流道中心高度
CH_R = 1.0        # 主流道半徑（2 mm）


def actuator_body(bore, mark_dots):
    body = cyl_z(0, 0, 0, BODY_H, BODY_R)
    # 頂面指壓凹槽（淺碟）
    dish = mf.Manifold.sphere(30, 128).translate([0, 0, BODY_H + 30 - 0.8])
    body = body - dish
    # 頂面識別圓點（1~4 點 = 1~4 號噴頭），並朝噴射方向排列當作箭頭
    for i in range(mark_dots):
        body = body - cyl_z(-3 + i * 2.2, 0, BODY_H - 1.4, 2, 0.6)
    return body


def actuator_cavity(bore):
    r = bore / 2
    sock = cyl_z(0, 0, -EPS, SOCKET_DEPTH + EPS, r)
    chamfer = cyl_z(0, 0, -EPS, 0.8, r + 0.5, r)            # 入口導角，好壓入
    riser = cyl_z(0, 0, SOCKET_DEPTH - EPS, CH_Z - SOCKET_DEPTH + EPS, 0.8)
    elbow = mf.Manifold.sphere(CH_R, 32).translate([0, 0, CH_Z])
    return sock + chamfer + riser + elbow


# ---------------------------------------------------------------- 1. 遠距


def long_range(bore, orifice):
    L = 38.0
    barrel = hull(flat_bottom_section(5, CH_Z, 4.5), flat_bottom_section(L - EPS, CH_Z, 3.0, 6.0))
    solid = actuator_body(bore, 1) + barrel
    ro = orifice / 2
    cav = actuator_cavity(bore)
    cav += capsule((0, 0, CH_Z), (L - 4.0, 0, CH_Z), CH_R)
    cav += cone_x(L - 4.0, L - 1.5, 0, CH_Z, CH_R, ro)        # 收斂段，加速成柱狀
    cav += cone_x(L - 1.6, L + 0.5, 0, CH_Z, ro, ro)          # 噴孔直段 L/D≈3
    return solid - cav


# ---------------------------------------------------------------- 2. 近距


def short_range(bore):
    L = 17.0
    barrel = hull(flat_bottom_section(5, CH_Z, 5.0), flat_bottom_section(L - EPS, CH_Z, 6.5))
    solid = actuator_body(bore, 2) + barrel
    cav = actuator_cavity(bore)
    cav += capsule((0, 0, CH_Z), (L - 6.0, 0, CH_Z), CH_R)
    cav += cone_x(L - 6.0, L - 5.0, 0, CH_Z, 0.6, 0.6)       # 1.2 mm 喉部
    cav += cone_x(L - 5.0, L + 0.3, 0, CH_Z, 0.6, 5.0)       # 擴散錐（~40°，可免支撐列印）
    return solid - cav


# ---------------------------------------------------------------- 3. 大面積（扇形）


def wide_area(bore, hole=0.7, n=7, spread=45.0):
    PX = 18.0          # 分配腔中心
    R = 10.0           # 扇頭半徑
    head = cyl_z(PX, 0, 0, 12.0, R)
    neck = hull(flat_bottom_section(5, CH_Z, 4.5), flat_bottom_section(PX, CH_Z, 4.5))
    solid = actuator_body(bore, 3) + neck + head
    cav = actuator_cavity(bore)
    cav += capsule((0, 0, CH_Z), (PX, 0, CH_Z), CH_R)
    cav += cyl_z(PX, 0, CH_Z - 1.5, 3.0, 3.0)                 # 分配腔（穩壓、平均分流）
    for a in np.linspace(-spread, spread, n):
        t = np.radians(a)
        d = np.array([np.cos(t), np.sin(t), 0.0])
        p0 = np.array([PX, 0, CH_Z]) + d * 2.5
        p1 = np.array([PX, 0, CH_Z]) + d * (R + 1.0)
        cav += capsule(p0, p1, hole / 2)
    return solid - cav


# ---------------------------------------------------------------- 4. 縫隙


def crevice(bore, tip_hole=1.2):
    L = 85.0           # 由按壓座中心算起，伸出約 75 mm
    TZ = 2.4           # 細管軸心高度
    TR = 2.4           # 細管外半徑（外徑 4.8 mm）
    trans = hull(flat_bottom_section(5, CH_Z, 4.5), flat_bottom_section(22, TZ, TR))
    tube = hull(flat_bottom_section(22, TZ, TR), flat_bottom_section(L - EPS, TZ, TR))
    solid = actuator_body(bore, 4) + trans + tube
    cav = actuator_cavity(bore)
    cav += capsule((0, 0, CH_Z), (9, 0, CH_Z), CH_R)
    cav += capsule((9, 0, CH_Z), (21, 0, TZ), 0.8)            # 下斜過渡
    cav += cone_x(21, L + 0.5, 0, TZ, tip_hole / 2, tip_hole / 2)
    return solid - cav


# ---------------------------------------------------------------- 量測規


def fit_gauge(sizes=(3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2)):
    pitch = 9.0
    w = pitch * len(sizes) + 4
    plate = box(0, w, 0, 14, 0, SOCKET_DEPTH + 1.5)
    for i, d in enumerate(sizes):
        x = 6.5 + i * pitch
        plate -= cyl_z(x, 6, -EPS, SOCKET_DEPTH + 2, d / 2)
        plate -= cyl_z(x, 6, -EPS, 0.8, d / 2 + 0.5, d / 2)
        # 刻度：孔下方刻 i+1 條短線，方便辨認（第 1 孔 = 3.0 mm）
        for k in range(i + 1):
            plate -= box(x - 3 + k * 1.0, x - 3 + k * 1.0 + 0.5, 11.2, 13.2,
                         SOCKET_DEPTH + 1.0, SOCKET_DEPTH + 2)
    return plate


# ---------------------------------------------------------------- 輸出


def save(man, path):
    parts = [p for p in man.decompose() if p.volume() > 1.0]   # 去除數值誤差碎片
    man = mf.Manifold.batch_boolean(parts, mf.OpType.Add) if len(parts) > 1 else parts[0]
    m = man.to_mesh()
    tm = trimesh.Trimesh(vertices=m.vert_properties[:, :3], faces=m.tri_verts, process=False)
    assert man.status() == mf.Error.NoError and tm.is_watertight, f"{path} 不是封閉實體"
    tm.export(path)
    bb = tm.bounds
    print(f"  {os.path.basename(path):22s} {bb[1][0]-bb[0][0]:5.1f} x {bb[1][1]-bb[0][1]:5.1f} x "
          f"{bb[1][2]-bb[0][2]:5.1f} mm   體積 {tm.volume/1000:5.2f} cm³  通孔數 {man.genus()}")
    return tm


def main():
    ap = argparse.ArgumentParser(description="RoachHunter 殺蟑噴頭 STL 產生器")
    ap.add_argument("--bore", type=float, default=3.4, help="閥桿套孔直徑 mm（預設 3.4）")
    ap.add_argument("--orifice", type=float, default=0.6, help="遠距噴孔直徑 mm（預設 0.6）")
    ap.add_argument("--out", default="stl", help="輸出資料夾")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(f"套孔 {a.bore} mm，遠距噴孔 {a.orifice} mm → {a.out}/")
    save(long_range(a.bore, a.orifice), f"{a.out}/1_long_range.stl")
    save(short_range(a.bore), f"{a.out}/2_short_range.stl")
    save(wide_area(a.bore), f"{a.out}/3_wide_area.stl")
    save(crevice(a.bore), f"{a.out}/4_crevice.stl")
    save(fit_gauge(), f"{a.out}/stem_fit_gauge.stl")


if __name__ == "__main__":
    main()
