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

model circuit(i1(t), i2(t), i3(t)) {
    let inputVoltage = sin(t) 
    use resistor(v = inputVoltage - v2)
    use resistor(v = v2)
    use capacitor(v = inputVoltage - v2)
    use kirchoff_current(i1 = i1, i2 = i2, i3 = i3)
}
