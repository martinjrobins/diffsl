RHS { 
    p * 2 + g * u[0],
    u[1],
}

F {
    dudt,
    0,
}

G {
    RHS,
    u[0] + u[1] + t,
}