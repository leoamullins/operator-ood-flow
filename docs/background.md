# Background and problem setup

[← Back to README](../README.md)

## Operator learning

Standard supervised learning methods map vectors to vectors (a feature vector to a target vector), in other words the method learns a function. Operator learning maps *functions* to *functions*; here, an input function $u$ to the output of some operator applied to it.

## Antiderivative operator (validation case)

$$
G: u \mapsto s, \qquad s(x) = \int_0^x u(t) \, dt, \qquad s(0) = 0
$$

This case is deliberately trivial as an application and quadrature solves it to machine precision. It is used to debug the implementation rather than as a difficult learning problem.

## Steady flow past a convex body

$$
G: \text{shape} \mapsto (u_x, u_y, p)
$$

The operator maps the geometry of a 2D body in a channel to the steady flow field around it at $Re = 20$. The shape is given to the network as its signed distance function (SDF), and the output is the velocity and pressure at arbitrary query points.

Next: [Data generation](data.md)
