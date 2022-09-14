model capacitor( i(t), v(t), c -> NonNegative) {
    i = c * dot(v)
}

model resistor( i(t), v(t), r -> NonNegative) {
    v = i * r
}

model inductor(l -> NonNegative, v(t), i(t)) {
    v = l * dot(i)
}

model kirchoff_current(i1(t), i2(t), i3(t)) {
    i1 = i2 + i3
}

model circuit(i1(t), i2(t), i3(t), v2(t)) {
    let inputVoltage = sin(t) 
    use resistor(v = inputVoltage - v2, i = i1, r = 1)
    use resistor(v = v2, i = i2, r = 1.5)
    use capacitor(v = inputVoltage - v2, i = i3, c = 2)
    use kirchoff_current(i1 = i1, i2 = i2, i3 = i3)
}
