RHS { 
    1 * 2 + 3 * u[0]
    u[1]
}

F {
    dudt[0]
    dudt[1]
    0
}

G {
    RHS[0]
    RHS[1]
    u[0] + u[1] + t 
}