# MFFineReason accepted snapshot review

N=26000, parts=260

## Difficulty
[
  [
    "easy",
    17056
  ],
  [
    "medium",
    8685
  ],
  [
    "hard",
    259
  ]
]

## Visual dependency
[
  [
    "strong",
    13665
  ],
  [
    "medium",
    9411
  ],
  [
    "weak",
    2924
  ]
]

## Source top
[
  [
    "MMR1",
    17947
  ],
  [
    "GameQA-140K",
    2235
  ],
  [
    "FineVision-visualwebinstruct(filtered)",
    1908
  ],
  [
    "BMMR",
    1274
  ],
  [
    "WaltonColdStart",
    548
  ],
  [
    "Euclid30K",
    454
  ],
  [
    "ViRL39K",
    425
  ],
  [
    "FineVision-raven",
    250
  ],
  [
    "MMK12",
    229
  ],
  [
    "FineVision-geo170k(qa)",
    107
  ],
  [
    "FineVision-geometry3k(mathv360k)",
    91
  ],
  [
    "Zebra-CoT-Physics",
    89
  ],
  [
    "LLaVA-CoT",
    69
  ],
  [
    "FineVision-tqa",
    57
  ],
  [
    "WeMath2-Pro",
    56
  ]
]

## Bench >=2 / >=3
{
  "ge2": {
    "MathVista": 24878,
    "MathVerse": 22124,
    "LogicVista": 10739,
    "GSM8K": 6332,
    "VideoHolmes": 14
  },
  "ge3": {
    "MathVista": 22761,
    "MathVerse": 19822,
    "LogicVista": 4570,
    "GSM8K": 949,
    "VideoHolmes": 4
  }
}

## Tags top
[
  [
    "geometry",
    14697
  ],
  [
    "visual_reasoning",
    4404
  ],
  [
    "chart_table",
    4142
  ],
  [
    "logic_puzzle",
    4045
  ],
  [
    "spatial_reasoning",
    3798
  ],
  [
    "arithmetic",
    3575
  ],
  [
    "coordinate_geometry",
    3553
  ],
  [
    "algebra",
    3401
  ],
  [
    "trigonometry",
    1496
  ],
  [
    "angle_chasing",
    1383
  ],
  [
    "area",
    1344
  ],
  [
    "word_problem",
    1134
  ],
  [
    "graph_interpretation",
    1071
  ],
  [
    "parallel_lines",
    1036
  ],
  [
    "grid_reasoning",
    1008
  ],
  [
    "right_triangle",
    983
  ],
  [
    "counting",
    903
  ],
  [
    "statistics",
    822
  ],
  [
    "flowchart",
    794
  ],
  [
    "proportional_reasoning",
    763
  ],
  [
    "similar_triangles",
    736
  ],
  [
    "3d_geometry",
    735
  ],
  [
    "data_interpretation",
    719
  ],
  [
    "proof",
    700
  ],
  [
    "physics",
    684
  ]
]

## Pass rate buckets
[
  [
    "0",
    12139
  ],
  [
    "1",
    10110
  ],
  [
    "(0.5,1)",
    1719
  ],
  [
    "(0.25,0.5]",
    1096
  ],
  [
    "(0,0.25]",
    936
  ]
]

## Samples


### 1. hard_low_pass id=543681 idx=7382 src=MMR1 diff=hard pass=0.0 tags=['geometry', 'analytic_geometry', 'conic_sections', 'optimization'] bench={'GSM8K': 0, 'LogicVista': 1, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In the diagram, circle O and ellipse T: \\(\\frac{x^2}{a^2}+\\frac{y^2}{b^2}=1\\), \\(a>b>0\\), meet at \\(M(0,1)\\). The ellipse has eccentricity \\(\\frac{\\sqrt3}{2}\\). Two perpendicular lines through M intersect the circle at A, B and the ellipse at C, D as shown. Find: (1) the equations of the ellipse and circle; (2) for any point P on the ellipse, the maximum of \\(d_1^2+d_2^2\\), where \\(d_1,d_2\\) are distances from P to the two perpendicular lines; (3) if \\(3\\overline{MA}\\cdot\\ove

A: \\(T:\\frac{x^2}{4}+y^2=1\\), \\(O:x^2+y^2=1\\); \\(\\max(d_1^2+d_2^2)=\\frac{16}{3}\\); the lines are \\(\\sqrt2x-y+1=0\\) and \\(x+\\sqrt2y-\\sqrt2=0\\).

Steps: Since \\(M(0,1)\\) is the top point of the ellipse, \\(b=1\\). With eccentricity \\(e=\\sqrt{1-b^2/a^2}=\\frac{\\sqrt3}{2}\\), get \\(a^2=4\\), so \\(T:\\frac{x^2}{4}+y^2=1\\). | Circle O is centered at the visible origin O and passes through \\(M(0,1)\\), so its radius is 1 and its equation is \\(x^2+y^2=1\\). | Because the two lines through M are perpendicular, for any point P, \\(d_1^2+d_2^2=MP^2\\).


### 2. hard_low_pass id=1192113 idx=2368 src=MMR1 diff=hard pass=0.0 tags=['geometry', '3d_spatial_reasoning', 'coordinate_geometry', 'folding'] bench={'GSM8K': 0, 'LogicVista': 1, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In trapezoid ABCD, AD is parallel to BC and ∠ABC = 90°. The side-length ratio AD:BC:AB is 2:3:4. Points E and F are the midpoints of AB and CD, respectively. Quadrilateral ADFE is folded along line EF. Four conclusions are given: ① DF ⊥ BC, ② BD ⊥ FC, ③ Plane DBF ⊥ Plane BFC, ④ Plane DCF ⊥ Plane BFC. Which conclusions may hold during the folding process? Fill in the conclusion numbers.

A: ②③

Steps: Set coordinates using the given ratio: B=(0,0,0), A=(0,4,0), C=(3,0,0), D=(2,4,0). Then E=(0,2,0) and F=(2.5,2,0). | Fold ADFE about EF. If the folding angle is θ, then D moves to D'=(2, 2+2cosθ, 2sinθ), while B, C, F stay fixed. | For ①, vector D'F=(0.5, -2cosθ, -2sinθ) and BC=(3,0,0). Their dot product is 1.5, never 0, so ① cannot hold.


### 3. hard_low_pass id=502680 idx=24813 src=MMR1 diff=hard pass=0.0 tags=['combinatorics', 'modular_arithmetic', 'sequence_sum', 'geometry'] bench={'GSM8K': 0, 'LogicVista': 2, 'MathVerse': 4, 'MathVista': 3, 'VideoHolmes': 0}

Q: Two concentric disks are divided into n equal sectors, with the numbers 1,2,...,n written sequentially on both disks in the initial aligned position. The inner disk can be rotated by one sector at a time. For each of the n rotational positions, define the rotation sum as the sum over all overlapping sectors of the product of the two numbers in that sector. (1) Find the sum of all n rotation sums. (2) If n is even, find the minimum rotation sum. (3) Let n=4m. In the initial aligned position, if a

A: (1) \(\left(\frac{n(n+1)}2\right)^2\). (2) For even \(n\), the minimum is \(\frac{n(n+2)(5n+2)}{24}\). (3) Mark the zero positions as a set \(A\) of size \(m\) in \(\mathbb Z_{4m}\). Since the cyclic difference set \(A-A\) has size at most \(1+m(m-1)<4m\) for \(m\le4\), choose a rotation shift outside \(A-A\); then no zero overlaps another zero.

Steps: Index the fixed outer sectors by \(1,2,\dots,n\). In the initial position the inner sectors are aligned with the same labels, as shown. | For a fixed outer sector label \(i\), as the inner disk takes all \(n\) rotations, the inner number overlapping it runs through \(1,2,\dots,n\) exactly once. | Thus the total of all rotation sums is \(\sum_{i=1}^n i\sum_{j=1}^n j=\left(\frac{n(n+1)}2\right)^2\).


### 4. hard_low_pass id=930611 idx=15630 src=MMR1 diff=hard pass=0.0 tags=['geometry', 'coordinate_geometry', 'circle', 'line_distance', 'right_triangle'] bench={'GSM8K': 0, 'LogicVista': 0, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In the Cartesian coordinate system, a circle has center A(1,0), and point M(4,4) lies on the circle. The line y = -3/4 x + b passes through M and intersects the x-axis and y-axis at B and C respectively. (1) Find the radius of the circle and b. (2) Determine the positional relationship between line BC and the circle, and justify it. (3) If P is on the circle and Q is on the y-axis below C, find all coordinates of Q such that triangle PQM is an isosceles right triangle.

A: Radius = 5, b = 7; line BC is tangent to the circle; Q = (0,2), (0,-8), (0,0), or (0, 3 - sqrt(41)).

Steps: The radius is AM = sqrt((4-1)^2 + (4-0)^2) = 5. | Since M(4,4) lies on y = -3/4 x + b, substitute to get 4 = -3 + b, so b = 7. | Thus BC has equation y = -3/4 x + 7, or 3x + 4y - 28 = 0.


### 5. hard_low_pass id=1340842 idx=4948 src=MMR1 diff=hard pass=0.0 tags=['coordinate_geometry', 'analytic_geometry', 'reflection', 'parabola', 'trapezoid'] bench={'GSM8K': 1, 'LogicVista': 1, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In the figure, the line y=3x+3 intersects the x-axis and y-axis at points B and A respectively. The parabola L: y=ax^2+bx+c has its vertex G on the x-axis and passes through (0,4) and (4,4). (1) Find the equation of the parabola L. (2) Does there exist a point C on the parabola L such that quadrilateral ABGC forms a trapezoid with BG as the base? If it exists, give the coordinates of C; otherwise explain why. (3) If parabola L is translated horizontally to obtain parabola L1 with vertex P, and t

A: (1) L: y=(x-2)^2. (2) Yes, C=(2+√3,3) or C=(2-√3,3). (3) Yes, the nondegenerate parabola is L1: y=(x+22/27)^2.

Steps: From y=3x+3, the intercepts are A=(0,3) and B=(-1,0). | Since L passes through (0,4) and (4,4), its axis of symmetry is x=2. Its vertex G lies on the x-axis, so G=(2,0). Thus L has form y=a(x-2)^2. | Using (0,4): 4=4a, so a=1. Hence L: y=(x-2)^2.


### 6. hard_low_pass id=753977 idx=1269 src=MMR1 diff=hard pass=0.0 tags=['geometry', 'analytic_geometry', 'optimization', 'algebra', 'visual_reasoning'] bench={'GSM8K': 0, 'LogicVista': 1, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: Given the ellipse E: x^2/a^2 + y^2/b^2 = 1 with a > b > 0. Its major axis is √2 times its minor axis, and the chord through a focus perpendicular to the x-axis has length 2√3. (I) Find the equation of E. (II) Point P moves on E with x-coordinate greater than 2. Points B and C lie on the y-axis, and the circle (x - 1)^2 + y^2 = 1 is inscribed in triangle PBC. Determine the position of P when the area S of triangle PBC is minimized, and justify the result.

A: E: x^2/12 + y^2/6 = 1. The minimum area occurs at P = (2√3, 0).

Steps: For E, the major and minor axis lengths are 2a and 2b. Since 2a = √2·2b, we have a = √2 b. | For a focus (c,0), c^2 = a^2 - b^2. The vertical focal chord x = c has length 2b^2/a. Thus 2b^2/a = 2√3, so b^2/a = √3. | Solving a = √2 b and b^2/a = √3 gives b = √6 and a = 2√3. Hence E is x^2/12 + y^2/6 = 1.


### 7. hard_low_pass id=1062480 idx=5403 src=MMR1 diff=hard pass=0.0 tags=['geometry', '3d_geometry', 'coordinate_geometry', 'spatial_reasoning'] bench={'GSM8K': 0, 'LogicVista': 0, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In the quadrilateral pyramid P-ABCD, PA ⟂ plane ABCD, AD ∥ BC, AD ⟂ CD, AD = CD = 2√2, BC = 4√2, and PA = 2. Point M lies on PD. (1) Prove that AB ⟂ PC. (2) If the dihedral angle M-AC-D is 45°, find the sine of the angle between line BM and plane PAC.

A: AB ⟂ PC; the required sine is 5√3/9.

Steps: Set coordinates using the base relations: A=(0,0,0), D=(2√2,0,0), C=(2√2,2√2,0), B=(-2√2,2√2,0), and since PA ⟂ base with PA=2, let P=(0,0,2). | Then AB=(-2√2,2√2,0) and PC=(2√2,2√2,-2). Their dot product is -8+8+0=0, so AB ⟂ PC. | Let M lie on PD as M=P+t(D-P)=(2√2t,0,2-2t), 0≤t≤1.


### 8. hard_low_pass id=687819 idx=11500 src=MMR1 diff=hard pass=0.0 tags=['geometry', 'proof', 'coordinate_geometry', 'angle_bisector', 'parallelogram'] bench={'GSM8K': 0, 'LogicVista': 1, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In parallelogram ABCD, point E is on AD. Triangle ABE is folded along BE to obtain triangle GBE, with point G inside parallelogram ABCD. Line BG is extended to intersect DC at point F, and EF bisects angle DEG. (1) Prove that GF = DF. (2) If BC = DC = 4DF, and the perimeter of quadrilateral BEFC is 14 + 6√5, find the length of BC.

A: GF = DF; BC = 8

Steps: Set E as the origin, take ED along the positive x-axis, and write A=(-a,0). Since EF bisects ∠DEG, let EF make angle -u with ED; then G=(a cos 2u,-a sin 2u) and F=(r cos u,-r sin u), where r=EF. | Because folding across BE reflects A to G, line BE is the angle bisector between EA and EG. Thus for b=BE, write B=(-b sin u,-b cos u). Let s=sin u and c=cos u. | Using the collinearity of B,G,F gives r(b-as)=abc and also GF/BG=as/(b-as).


### 9. medium id=516616 idx=13079 src=MMR1 diff=medium pass=0.75 tags=['geometry', 'trigonometry', 'optimization', 'visual_reasoning'] bench={'GSM8K': 1, 'LogicVista': 0, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: A sector of a circle has central angle 60° and radius 1 m. Two inscribed rectangles are cut as shown: in method A, one side of the rectangle lies on radius OA; in method B, one side of the rectangle is parallel to chord AB. (1) For method A, let ∠NOA = θ. Find the rectangle area as a function of θ and its maximum value. (2) Which method gives the larger maximum area? Explain.

A: Method A: S(θ)=sinθ(cosθ−sinθ/√3), maximum √3/6 m² at θ=30°. Method B has maximum 2−√3 m², so method A gives the larger maximum area.

Steps: Set O=(0,0), OA on the x-axis, radius 1, so OB has equation y=√3x and the arc is x²+y²=1. | For method A, N=(cosθ,sinθ). Since MN is perpendicular to OA, M=(cosθ,0). Since P lies on OB at height sinθ, P=(sinθ/√3,sinθ). | Thus the rectangle height is sinθ and its base is cosθ−sinθ/√3, so S_A(θ)=sinθ(cosθ−sinθ/√3), 0≤θ≤π/3.


### 10. medium id=3905 idx=4808 src=MMR1 diff=medium pass=0.5 tags=['geometry', 'spatial_reasoning', 'rotation'] bench={'GSM8K': 0, 'LogicVista': 2, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: Two congruent squares ABCD and CDEF share side CD as shown. Square ABCD can be rotated to coincide with square CDEF. How many points in the plane can serve as the center of such a rotation?

A: 3

Steps: The figure shows two congruent adjacent squares sharing the vertical side CD: ABCD is on the right and CDEF is on the left. | A 90° counterclockwise rotation about C maps the right square ABCD onto the left square CDEF. | A 90° clockwise rotation about D also maps ABCD onto CDEF.


### 11. medium id=889515 idx=10516 src=MMR1 diff=medium pass=0.75 tags=['geometry', 'circle_theorems', 'trigonometry', 'proof'] bench={'GSM8K': 1, 'LogicVista': 2, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In the figure, AB is the diameter of circle O, BC is tangent to circle O at B, AC intersects the circle again at E, and D lies on AC such that ∠AOD = ∠C. (1) Prove that OD ⟂ AC. (2) If AE = 8 and tan A = 3/4, find OD.

A: OD ⟂ AC, and OD = 3

Steps: Since BC is tangent at B and OB lies on diameter AB, ∠ABC = 90°. | Thus in right triangle ABC, ∠A + ∠C = 90°. | Because D lies on AC and O lies on AB, ∠OAD = ∠A. Given ∠AOD = ∠C, triangle AOD has ∠ADO = 180° − ∠A − ∠C = 90°.


### 12. medium id=950098 idx=2258 src=MMR1 diff=medium pass=0.0 tags=['geometry', 'rotation', 'geometric_construction', 'spatial_reasoning'] bench={'GSM8K': 0, 'LogicVista': 1, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In the figure, triangles ABC and CDE are congruent right triangles, with ∠B = ∠CDE = 90°, and triangle CDE is obtained by rotating triangle ABC counterclockwise. Find the rotation center O by ruler-and-compass construction and state the rotation angle.

A: O is the intersection of the perpendicular bisectors of AC and BD; the rotation angle is 90° counterclockwise.

Steps: From the ordered congruence and the right-angle vertices, the rotation sends A→C, B→D, and C→E. | The rotation center must be equidistant from each point and its image, so it lies on the perpendicular bisectors of AC, BD, and CE. | Constructing the perpendicular bisectors of AC and BD gives their intersection O, the rotation center.


### 13. medium id=1420776 idx=5984 src=MMR1 diff=medium pass=0.0 tags=['chart_table', 'statistics', 'proportional_reasoning', 'probability', 'geometry'] bench={'GSM8K': 2, 'LogicVista': 2, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: On the eve of Arbor Day, all students in a school planted between 2 and 6 trees. The image shows an incomplete pie chart and an incomplete bar chart of the survey results. Answer: (1) What is the total number of students? (2) Complete the missing bars in the bar chart. (3) What is the central angle of the pie-chart sector for students who planted 3 trees? (4) What are the mode and median of the numbers of trees planted? (5) What is the probability that a randomly selected student planted 6 trees

A: (1) 1000 students; (2) missing bars: 2 trees = 100 students, 5 trees = 350 students; (3) 72°; (4) mode = 5 trees, median = 4 trees; (5) 0.05 = 5%

Steps: From the bar chart, 4 trees corresponds to 300 students; from the pie chart, 4 trees is 30%, so the total is 300 ÷ 0.30 = 1000 students. | The missing bar for 2 trees is 10% of 1000, so it is 100 students. | The missing bar for 5 trees is 35% of 1000, so it is 350 students.


### 14. medium id=848072 idx=9621 src=MMR1 diff=medium pass=0.75 tags=['geometry', 'coordinate_geometry', 'angle_reasoning', 'length_ratio'] bench={'GSM8K': 0, 'LogicVista': 1, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In triangle ABC, ∠ACB = 90°, AC = BC, D is the midpoint of BC, and CF ⟂ AD. From the diagram, F lies on AB. The following conclusions are given: ① ∠ADF = 45°; ② ∠ADC = ∠BDF; ③ AF = 2BF; ④ CF = 3DF. How many of these conclusions are correct? A. 1 B. 2 C. 3 D. 4

A: B. 2

Steps: Set coordinates C=(0,0), A=(0,a), B=(a,0), since triangle ABC is an isosceles right triangle at C. | D is the midpoint of BC, so D=(a/2,0). Line AD has slope -2, so the line CF perpendicular to AD through C has slope 1/2. | Since F lies on AB, solve y=x/2 with AB: y=-x+a, giving F=(2a/3,a/3).


### 15. easy_pass1 id=181606 idx=4373 src=FineVision-visualwebinstruct(filtered) diff=easy pass=1.0 tags=['geometry', 'similar_triangles', 'rectangle', 'proportional_reasoning'] bench={'GSM8K': 1, 'LogicVista': 1, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: As shown in the figure, ABCD is a rectangle with AB = 6.0 and BC = 8.0. Point E lies on diagonal BD with BE = 6.0. Line AE is extended to meet side DC at F. Find CF.

A: 2

Steps: Since ABCD is a rectangle, diagonal BD has length sqrt(6^2 + 8^2) = 10. | Given BE = 6, we have ED = 10 - 6 = 4, so BE:ED = 3:2. | Because AB is parallel to DC, triangles ABE and FDE are similar.


### 16. easy_pass1 id=943341 idx=14761 src=MMR1 diff=easy pass=1.0 tags=['geometry', 'trapezoid', 'right_triangle', 'pythagorean_theorem', 'spatial_reasoning'] bench={'GSM8K': 2, 'LogicVista': 2, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In the right trapezoid ABCD, where AB is parallel to DC and AD is perpendicular to DC at D, if AB = 1, AD = 2, and DC = 4, what is the length of BC?

A: \sqrt{13}

Steps: Since AD is perpendicular to DC and AB is parallel to DC, AD is the height of the trapezoid, so the vertical difference from B to C is 2. | The top base AB has length 1 and the bottom base DC has length 4, so the horizontal difference from B to C is 4 - 1 = 3. | Thus BC is the hypotenuse of a right triangle with legs 2 and 3.


### 17. easy_pass1 id=1421747 idx=21012 src=MMR1 diff=easy pass=1.0 tags=['geometry', 'area_reasoning', 'proof', 'perpendicular_distances'] bench={'GSM8K': 1, 'LogicVista': 2, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In the figure, P is inside equilateral triangle ABC. Perpendiculars from P meet AB, AC, and BC at E, F, and D respectively, and AH is perpendicular to BC. Prove using the triangle area formula that PE + PF + PD = AH.

A: PE + PF + PD = AH.

Steps: Let the common side length of the equilateral triangle be s, so AB = BC = AC = s. | Using base BC and height AH, the area of triangle ABC is (1/2)s·AH. | Since P lies inside ABC, triangles PAB, PBC, and PCA partition triangle ABC.


### 18. easy_pass1 id=1319162 idx=9176 src=MMR1 diff=easy pass=1.0 tags=['geometry', 'rotation', 'angle_reasoning'] bench={'GSM8K': 1, 'LogicVista': 2, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: Triangle ABC is rotated 80° counterclockwise around point A to obtain triangle AB′C′. If ∠BAC = 50°, then what is the measure of ∠CAB′?

A: 30°

Steps: A rotation about point A sends ray AB to ray AB′, so ∠BAB′ equals the rotation angle, 80°. | The given angle ∠BAC is 50°. | From the diagram, ray AC lies between rays AB and AB′ at point A.


### 19. easy_pass1 id=794197 idx=11419 src=MMR1 diff=easy pass=1.0 tags=['geometry', 'circle_theorem', 'inscribed_angle', 'visual_geometry'] bench={'GSM8K': 0, 'LogicVista': 1, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In the figure, ⊙O is the circumcircle of triangle ABD, AB is a diameter of ⊙O, CD is a chord of ⊙O, and ∠ABD = 58°. Find the degree measure of ∠BCD.

A: 32°

Steps: Since AB is a diameter of the circle, the inscribed angle ∠ADB subtending AB is 90°. | In triangle ABD, ∠BAD = 180° − 90° − 58° = 32°. | Angles ∠BAD and ∠BCD are inscribed angles subtending the same chord BD.


### 20. easy_pass1 id=779528 idx=5451 src=MMR1 diff=easy pass=1.0 tags=['geometry', 'rotational_symmetry', 'visual_reasoning'] bench={'GSM8K': 0, 'LogicVista': 2, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: Which of the labeled shapes can coincide with itself after rotating 60 degrees around its center?

A: A

Steps: A shape that matches itself after a 60° rotation must have rotational symmetry of order 6, since 360° ÷ 60° = 6. | Option A is a hexagon, which has 6-fold rotational symmetry, so a 60° rotation maps it onto itself. | Options B, C, and D are a pentagon, square, and triangle, whose basic rotational symmetry angles are 72°, 90°, and 120°, respectively.


### 21. tag_geometry id=1090984 idx=157 src=MMR1 diff=easy pass=0.25 tags=['geometry', 'angle_reasoning'] bench={'GSM8K': 0, 'LogicVista': 1, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In the figure, if ∠1 = ∠2, then both angles are ___°. If ∠3 = ∠4 = ∠5, then each angle is ___°. What is ∠1 + ∠2 + ∠3 + ∠4 + ∠5?

A: 45°, 60°, 270°

Steps: The right-angle marker shows the vertical and horizontal lines are perpendicular, so each quadrant formed by them is 90°. | Angles 1 and 2 split the upper-right quadrant, so ∠1 + ∠2 = 90°. | If ∠1 = ∠2, then each is 90° ÷ 2 = 45°.


### 22. tag_geometry id=54999 idx=17519 src=BMMR diff=easy pass=0.0 tags=['geometry', 'spatial_reasoning', 'net_to_solid_matching'] bench={'GSM8K': 0, 'LogicVista': 1, 'MathVerse': 3, 'MathVista': 4, 'VideoHolmes': 0}

Q: The four shown diagrams are nets of known 3D shapes. Match them from left to right to: (A) Cylinder, (B) Cube, (C) Triangular Prism, (D) Square Pyramid.

A: B, A, C, D

Steps: The first net is made of six equal squares, which fold to form a cube. | The second net has one rectangle and two circles, matching a cylinder's curved side and two circular bases. | The third net has three rectangles and two triangles, matching a triangular prism.


### 23. tag_geometry id=876998 idx=30789 src=MMR1 diff=easy pass=0.0 tags=['geometry', 'algebraic_reasoning'] bench={'GSM8K': 2, 'LogicVista': 1, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In the isosceles triangle ABC shown, BD is a median to the equal side AC. The median divides the perimeter into two parts of lengths 12 cm and 9 cm. Find the possible side lengths of the triangle.

A: 8 cm, 8 cm, 5 cm or 6 cm, 6 cm, 9 cm

Steps: Let the equal sides have length a, and let the base have length b. | Since BD is a median to the equal side AC, point D is the midpoint of AC, so AD = DC = a/2. | The two perimeter parts cut off by BD are AB + AD = a + a/2 = 3a/2 and BC + CD = b + a/2.


### 24. tag_geometry id=67823 idx=19431 src=MMR1 diff=easy pass=0.25 tags=['geometry', 'logic_puzzle', 'pattern_recognition', 'sequence'] bench={'GSM8K': 1, 'LogicVista': 4, 'MathVerse': 3, 'MathVista': 4, 'VideoHolmes': 0}

Q: Observe the pattern of white triangles inside the first three large triangles. How many white triangles are in the 5th large triangle?

A: 121

Steps: From the figures, the 1st large triangle has 1 white triangle. | The 2nd large triangle keeps that white triangle and adds 3 smaller white triangles, for a total of 4. | The 3rd large triangle adds 9 still smaller white triangles, giving 1 + 3 + 9 = 13.


### 25. tag_chart_table id=465189 idx=6680 src=MMR1 diff=easy pass=0.0 tags=['chart_table', 'arithmetic', 'percentage', 'data_interpretation'] bench={'GSM8K': 2, 'LogicVista': 1, 'MathVerse': 3, 'MathVista': 4, 'VideoHolmes': 0}

Q: A school has 800 total participants in four activities: essay writing, solo singing, painting, and hand-copied newspapers. The charts show essay writing has 296 participants and 37%, solo singing has 80 participants, hand-copied newspapers has 224 participants and 28%, and painting accounts for 25%. (1) How many students participated in painting, and what is the central angle of the solo singing sector in the pie chart? (2) If the funding standards are 10 yuan, 12 yuan, 15 yuan, and 20 yuan per 

A: Painting: 200 students; solo singing angle: 36°; total funding: 11400 yuan.

Steps: Painting accounts for 25% of the total 800 participants, so painting participants = 800 × 25% = 200. | The known pie chart percentages are 37%, 28%, and 25%, so solo singing is 100% - 37% - 28% - 25% = 10%. | The central angle for solo singing is 10% × 360° = 36°.


### 26. tag_chart_table id=1590161 idx=23921 src=MMR1 diff=easy pass=0.0 tags=['chart_table', 'arithmetic', 'percentages', 'visual_reasoning'] bench={'GSM8K': 2, 'LogicVista': 1, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: A school surveyed students about which of Taiyuan City's core values they were most interested in: 包容, 尚德, 守法, 诚信, 卓越. Based on the bar chart and pie chart, answer: (1) How many students were surveyed in total? (2) Complete the missing bar in the bar chart and give the percentages for all categories in the pie chart.

A: 500 students; missing 尚德 bar = 100 students. Pie chart: 包容 30%, 尚德 20%, 守法 10%, 诚信 25%, 卓越 15%.

Steps: From the bar chart, 包容 has 150 students, and the pie chart labels 包容 as 30%. | So the total number surveyed is 150 ÷ 30% = 150 ÷ 0.3 = 500. | The visible bar values are 包容 150, 守法 50, 诚信 125, and 卓越 75; their sum is 400.


### 27. tag_chart_table id=1444475 idx=22589 src=MMR1 diff=easy pass=1.0 tags=['chart_table', 'proportion', 'visual_reasoning'] bench={'GSM8K': 0, 'LogicVista': 1, 'MathVerse': 2, 'MathVista': 3, 'VideoHolmes': 0}

Q: Is the majority of the pie chart colored Dark Green?

A: Yes

Steps: The legend identifies the dark green slice as "Dark Green" and the teal slice as "Light Seafoam." | Visually, the dark green slice covers more than half of the circle. | Since a majority means more than 50%, the Dark Green portion is the majority.


### 28. tag_chart_table id=1447522 idx=323 src=MMR1 diff=easy pass=0.0 tags=['chart_table', 'proportional_reasoning', 'arithmetic'] bench={'GSM8K': 1, 'LogicVista': 1, 'MathVerse': 2, 'MathVista': 4, 'VideoHolmes': 0}

Q: What percentage of the total does Dark Green represent, and how much more is it compared to Light Gold?

A: Dark Green represents 90% of the total and is 80 percentage points more than Light Gold.

Steps: The legend maps Light Gold to the yellow slice and Dark Green to the green slice. | The yellow Light Gold slice is about one-tenth of the pie, so it represents 10%. | A full pie is 100%, so Dark Green represents 100% - 10% = 90%.


### 29. tag_logic_puzzle id=10325 idx=13764 src=MMR1 diff=medium pass=0.0 tags=['logic_puzzle', 'constraint_satisfaction', 'planning'] bench={'GSM8K': 0, 'LogicVista': 4, 'MathVerse': 0, 'MathVista': 0, 'VideoHolmes': 0}

Q: A hunter must ferry a basket of carrots, a dog, two wolves, and a sheep across a river. The boat carries the hunter and up to two items. If the hunter is absent: wolves with the sheep without the dog is unsafe, the dog and sheep quarrel, and the sheep ruins the carrots. What should be taken first, and what full strategy gets everyone across safely?

A: Take the sheep first with one other item, e.g. sheep + carrots. Then bring the carrots back. A valid sequence is: S+C over, C back; W1+W2 over, S back; C+D over, hunter back alone; S over.

Steps: The sheep is the dangerous item: it cannot be left unattended with wolves, the dog, or carrots. | First take the sheep with one item, for example the carrots, across; then bring the carrots back so the sheep is alone on the far bank. | Take both wolves across together; bring the sheep back, leaving only the wolves on the far bank.


### 30. tag_logic_puzzle id=33376 idx=14886 src=GameQA-140K diff=easy pass=0.5 tags=['chart_table', 'logic_puzzle', 'spatial_reasoning', 'grid_search'] bench={'GSM8K': 0, 'LogicVista': 3, 'MathVerse': 0, 'MathVista': 1, 'VideoHolmes': 0}

Q: Rules: This is a word search game. Words can be placed in any of eight directions: right, down, diagonal-right-down, diagonal-right-up, diagonal-left-down, diagonal-left-up, up, or left. Words read from start to end in the specified direction.  Find the word "DELL" in the grid. Where does it start and in which direction does it go?

A: Row 2, Column 4, direction left

Steps: The word must start at a cell containing D. | The visible D cells are at row 2, column 4 and row 8, column 7. | From row 2, column 4, moving left gives row 2 column 3 = E, row 2 column 2 = L, and row 2 column 1 = L.


### 31. tag_logic_puzzle id=19636 idx=8018 src=GameQA-140K diff=medium pass=0.0 tags=['chart_table', 'logic_puzzle', 'cellular_automaton', 'spatial_reasoning'] bench={'GSM8K': 0, 'LogicVista': 4, 'MathVerse': 1, 'MathVista': 2, 'VideoHolmes': 0}

Q: In the 4×4 Game of Life grid, black cells are alive and white cells are dead. Consider the 3×3 region starting at cell (2,3), wrapping around the 4×4 grid to select its initial cells. Treat this 3×3 region as an independent toroidal Game of Life system. How many iterations are needed until the region is stable or repeating?

A: 0 iterations

Steps: The visible live cells in the 4×4 grid are (2,3), (3,0), (3,1), and (3,3). | The 3×3 region starting at (2,3) wraps to rows 2, 3, 0 and columns 3, 0, 1. | So the extracted 3×3 state is [[1,0,0],[1,1,1],[0,0,0]], with 4 live cells total.


### 32. tag_logic_puzzle id=99658 idx=10693 src=GameQA-140K diff=easy pass=0.0 tags=['logic_puzzle', 'game_rules', 'spatial_reasoning', 'counting'] bench={'GSM8K': 0, 'LogicVista': 4, 'MathVerse': 1, 'MathVista': 2, 'VideoHolmes': 0}

Q: In Ultra TicTacToe, after an opponent moves at coordinates (i, j, row, col), your next move must be in the Nine-grid (row, col), and you may choose any empty cell there. The opponent just placed a piece at (3, 2, 1, 1). How many possible coordinates are available for your next move?

A: 7

Steps: The opponent's move is (3, 2, 1, 1), so its internal cell coordinate is (row, col) = (1, 1). | By the placement rule, the next move must be in Nine-grid (1, 1). | In the image, Nine-grid (1, 1) is the top-left 3×3 grid.


### 33. tag_algebra id=1354383 idx=6160 src=MMR1 diff=medium pass=1.0 tags=['algebra', 'inequalities', 'optimization', 'rational_function'] bench={'GSM8K': 1, 'LogicVista': 1, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: Let x and y satisfy the constraints 2x + y - 2 >= 0, x - 2y + 4 >= 0, and x - 1 < 0. Find the maximum value of z = (x + y + 3)/(x + 3).

A: 5/3

Steps: Rewrite the constraints as y >= 2 - 2x, y <= (x + 4)/2, and x < 1. | For feasible y values to exist, need 2 - 2x <= (x + 4)/2, which gives x >= 0. Hence 0 <= x < 1. | Since z = (x + y + 3)/(x + 3) = 1 + y/(x + 3) and x + 3 > 0, z increases as y increases for fixed x.


### 34. tag_algebra id=43349 idx=2245 src=FineVision-visualwebinstruct(filtered) diff=easy pass=0.0 tags=['geometry', 'algebra', 'function_interpretation', 'graph_reading'] bench={'GSM8K': 1, 'LogicVista': 1, 'MathVerse': 3, 'MathVista': 4, 'VideoHolmes': 0}

Q: What equation is shown for the parabolic hill, and how does the height change as x moves along the x-axis?

A: The hill is modeled by y = 20(1 - x^2/6400). It is a downward-opening parabola: y is 20 at x = 0 and decreases quadratically to 0 at x = 80 m, with the same behavior symmetrically at x = -80 m.

Steps: Read the equation labeled beside the curve: y = 20(1 - x^2/6400). | At x = 0, the equation gives y = 20, so the peak is at height 20. | The negative x^2 term means the parabola opens downward, so y decreases as |x| increases.


### 35. tag_algebra id=650741 idx=22372 src=MMR1 diff=medium pass=1.0 tags=['geometry', 'coordinate_geometry', 'optimization', 'algebra'] bench={'GSM8K': 1, 'LogicVista': 2, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In the figure, A(5,n), B(0,4), where n>0. Point P moves from the origin O to the right along the x-axis at a speed of 1 unit per second. Connect AP and draw ray PQ perpendicular to AP, intersecting the y-axis at Q. Let the time P has moved be t seconds, t>0. (1) When does Q coincide with O? (2) If n=2, during the movement of P, what is the shortest distance from Q to B?

A: 5, 7/8

Steps: Since P moves right from O at speed 1, after t seconds P=(t,0). Let Q=(0,q) on the y-axis. | From the perpendicular relation AP ⟂ PQ, use vectors AP=(5-t,n) and PQ=(-t,q). Their dot product is 0. | (5-t)(-t)+nq=0, so q=(5t-t^2)/n.


### 36. tag_algebra id=416725 idx=19927 src=MMR1 diff=easy pass=0.75 tags=['coordinate_geometry', 'functions', 'algebra', 'graph_interpretation', 'area'] bench={'GSM8K': 2, 'LogicVista': 1, 'MathVerse': 4, 'MathVista': 4, 'VideoHolmes': 0}

Q: In the figure, the inverse proportional function y = k/x intersects the linear function y = x + b at point A(1, -k + 4) in the first quadrant. (1) Determine the expressions of the two functions. (2) Connect OA and OB, where O is the origin and B is the other intersection point, and find the area of triangle AOB.

A: y = 2/x, y = x + 1; area = 3/2

Steps: Since A(1, -k + 4) lies on y = k/x, substitute x = 1: k = -k + 4, so k = 2. | Thus A = (1, 2). Since A also lies on y = x + b, substitute: 2 = 1 + b, so b = 1. | The two functions are y = 2/x and y = x + 1.
