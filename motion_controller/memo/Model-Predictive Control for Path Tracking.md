# Model-Predictive Control for Path Tracking

## Bicycle Kinematic Model
$$
\begin{equation}
\left\{
\begin{aligned}
\dot{x} = v\cos(\theta+\beta) \\
\dot{y} = v\sin(\theta+\beta) \\
\dot{\theta} = \frac{v}{l_r}\sin{\beta} \\
\dot{v} = a \\
\beta = \arctan({\frac{l_r}{l_f+l_r}\tan{\delta}})
\end{aligned}
\right.
\end{equation}
$$

Where
$$
x: \text{x coordinate}\\
y: \text{y coordinate}\\
l_f: \text{distance from the center of the mass of the vehicle to the front axles} \\
l_r: \text{distance from the center of the mass of the vehicle to the rear axles}\\
\beta: \text{angle of the current velocity  of the center of mass respected to x axis}\\
\theta: \text{heading} \\
a: \text{acceleration} \\
\delta: \text{steering angle}
$$
Reference: [Kinematic and dynamic vehicle models for autonomous driving control design](https://ieeexplore.ieee.org/document/7225830)
## MPC formulation
*State*

***
$$
\bf{z} = (x, y, v, \theta)^T \\
z \in \bf{R}^{4x1}
$$



*Control Input*
***
$$
\bf{u} = (a, \delta)^T \\
u \in \bf{R}^{2x1}
$$



Linearization around equilibrium points
$$
(\bar{\bf{z}},\bar{\bf{u}})
$$
***
$$
\newcommand\at[2]{\left.#1\right|_{#2}}

\dot{\bf{z}}=f(\bf{z},\bf{u}) = A^{'}\bf{z}+B^{'}\bf{u} \\
A^{'}=\at{\frac{\partial f}{\partial z}}{\bf{z}=\bar{\bf{z}}, \bf{u}=\bar{\bf{u}}}\\
B^{'}=\at{\frac{\partial f}{\partial u}}{\bf{z}=\bar{\bf{z}}, \bf{u}=\bar{\bf{u}}}\\
$$

Therefore
$$
A^{'} =
\at{\begin{pmatrix}
\frac{\partial}{\partial x}{v\cos{(\theta+\beta})} & \frac{\partial}{\partial y}{v\cos{(\theta+\beta})} & \frac{\partial}{\partial v}{v\cos{(\theta+\beta})} & \frac{\partial}{\partial \theta}{v\cos{(\theta+\beta})} \\
\frac{\partial}{\partial x}{v\sin{(\theta+\beta})} & \frac{\partial}{\partial y}{v\sin{(\theta+\beta})} & \frac{\partial}{\partial v}{v\sin{(\theta+\beta})} & \frac{\partial}{\partial \theta}{v\sin{(\theta+\beta})} \\
\frac{\partial}{\partial x}{a} & \frac{\partial}{\partial y}{a} & \frac{\partial}{\partial v}{a} & \frac{\partial}{\partial \theta}{a} \\
\frac{\partial}{\partial x}{\frac{v\sin{\beta}}{l_r}} & \frac{\partial}{\partial y}{\frac{v\sin{\beta}}{l_r}} & \frac{\partial}{\partial v}{\frac{v\sin{\beta}}{l_r}} & \frac{\partial}{\partial \theta}{\frac{v\sin{\beta}}{l_r}} \\
\end{pmatrix}}{\bf{z}=\bar{\bf{z}}, \bf{u}=\bar{\bf{u}}} \\

A^{'} = \begin{pmatrix}
0 & 0 & \cos{(\bar\theta+\bar\beta}) & {-\bar{v}\sin{(\bar\theta+\bar\beta})} \\
0 & 0 & {\sin{(\bar\theta+\bar\beta})} & {\bar{v}\cos{(\bar\theta+\bar\beta})} \\
0 & 0 & 0 & 0 \\
0 & 0 & {\frac{\sin{\bar\beta}}{l_r}} & 0 \\
\end{pmatrix}
$$

And
$$
B^{'} =
\at{\begin{pmatrix}
\frac{\partial}{\partial a}{v\cos{(\theta+\beta})} & \frac{\partial}{\partial \delta}{v\cos{(\theta+\beta})} \\
\frac{\partial}{\partial a}{v\sin{(\theta+\beta})} & \frac{\partial}{\partial \delta}{v\sin{(\theta+\beta})} \\
\frac{\partial}{\partial a}{a} & \frac{\partial}{\partial \delta}{a} \\
\frac{\partial}{\partial a}{\frac{v\sin{\beta}}{l_r}} & \frac{\partial}{\partial \delta}{\frac{v\sin{\beta}}{l_r}} \\
\end{pmatrix}}{\bf{z}=\bar{\bf{z}}, \bf{u}=\bar{\bf{u}}} \\

B^{'} =
\begin{pmatrix}
0 & 0 \\
0 & 0 \\
1 & 0 \\
0 & \frac{\bar{v}\cos{\bar\beta}}{l_f+l_r}*\frac{1}{1 + (\frac{l_r}{l_f+l_r}\tan{\bar\delta})^2}*\frac{1}{\cos^2\bar\delta} \\
\end{pmatrix}
$$



Discrete-time mode with Forward Euler Discretization with sampling time dt
***
$$
\bf{z}_{k+1} = \bf{z}_{k} + f(\bf{z}_k,\bf{u}_k)dt
$$

Applying first degree Taylor expansion on f around equilibrium point, we get
$$
f(\bf{z}_k, u_k) = f(\bar{z}, \bar{u}) + A^{'} (z_k - \bar{z}) + B^{'} (u_k - \bar{u})
$$

So for
$$
\bf{z}_{k+1} = A\bf{z}_{k} + B\bf{u}_{k} + C
$$
We have:
$$
A = (I + A^{'} dt)
 = \begin{pmatrix}
1 & 0 & \cos{(\bar\theta+\bar\beta})dt & {-v\sin{(\bar\theta+\bar\beta})}dt \\
0 & 1 & {\sin{(\bar\theta+\bar\beta})} & {\bar{v}\cos{(\bar\theta+\bar\beta})} \\
0 & 0 & 1 & 0 \\
0 & 0 & {\frac{\sin{\bar\beta}}{l_r}}dt & 1 \\
\end{pmatrix}
$$

$$
B = B^{'} dt
  = \begin{pmatrix}
0 & 0 \\
0 & 0 \\
dt & 0 \\
0 & \frac{\bar{v}\cos{\bar\beta}}{l_f+l_r}*\frac{1}{1 + (\frac{l_r}{l_f+l_r}\tan{\bar\delta})^2}*\frac{1}{\cos^2\bar\delta}dt \\
\end{pmatrix}
$$

$$
C = (f(\bar{z}, \bar{u}) - A^{'}\bar{z} - B^{'}\bar{u})dt
= dt (
\begin{pmatrix}
\bar{v}\cos{(\bar{\theta}+\bar{\beta})} \\
\bar{v}\sin{(\bar{\theta}+\bar{\beta})} \\
\bar{a} \\
\frac{\bar{v}\sin{\bar\beta}}{l_r} \\
\end{pmatrix} -

\begin{pmatrix}
\bar{v}\cos{(\bar{\theta}+\bar{\beta}) - \bar{v}\sin(\bar\theta+\bar\beta)}\bar\theta \\
\bar{v}\sin{(\bar{\theta}+\bar{\beta}) + \bar{v}\cos(\bar\theta+\bar\beta)}\bar\theta \\
0 \\
\frac{\bar{v}\sin{\bar\beta}}{l_r} \\
\end{pmatrix}
 -
\\
\begin{pmatrix}
0 \\
0 \\
\bar{a} \\
\frac{\bar{v}\cos{\bar\beta}}{l_f+l_r}*\frac{1}{1 + (\frac{l_r}{l_f+l_r}\tan{\bar\delta})^2}*\frac{1}{\cos^2\bar\delta} *\bar\delta \\
\end{pmatrix}
) =

dt(\begin{pmatrix}
\bar{v}\sin(\bar\theta+\bar\beta)\bar\theta \\
-\bar{v}\cos(\bar\theta+\bar\beta)\bar\theta \\
0 \\
-\frac{\bar{v}\cos{\bar\beta}}{l_f+l_r}*\frac{1}{1 + (\frac{l_r}{l_f+l_r}\tan{\bar\delta})^2}*\frac{1}{\cos^2\bar\delta} *\bar\delta \\
\end{pmatrix})
$$



*Path Tracking Problem*

***
$$
\newcommand{\argmin}{\mathop{\rm arg~min}\limits}

\argmin_{u_k} \sum_{k=0}^{N-1} ((z_k - z_{k,ref})^{T}Q(z_k - z_{k,ref}) + u_k^{T}Ru_k) + \\
(z_N - z_{N,ref})^TQ_{f}(z_N - z_{N,ref}) + \\
\sum_{k=0}^{N-1} (u_{k+1}- u_{k})^{T}R_{d}(u_{k+1}- u_{k})
$$

Subject to

$$
\text{given initial state } z_0 \text{and} \\
{z}_{k+1} = A{z}_{k} + B{u}_{k} + C (k = 0,...,N-1) \\
\frac{|u_{k+1} - u_{k}|}{dt} < u_{max} \\
u_{min} < u_k < u_{max} \\
v_{min} < v_k < v_{max} \\
$$

Where
$$
N: \text{horizon length} \\
z_{k, ref}: \text{reference state at k}\\
Q: \text{state cost matrix}\\
Qf: \text{state final matrix}\\
R: \text{input cost matrix}\\
Rd: \text{input difference cost matrix} \\
dt: \text{sampling time}
$$
