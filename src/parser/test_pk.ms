model peripheral(c(t), p(t)) {
   dot(c) += k * (p - c)
   dot(p) += -k * (p - c)
}

model pk_model(c(t), p1(t), p2(t), d(t), cl -> NonNegative) {
    use peripheral(c=c, p=p1)
    use peripheral(c=c, p=p2)
    dot(c) += -cl * c
    dot(p) += d
}
