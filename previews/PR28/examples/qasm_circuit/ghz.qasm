// GHZ state on six qubits, in the shape of a Qiskit `QuantumCircuit.qasm()` export.
// The trailing `measure meas -> q;` lines such an export ends with have been stripped:
// Canopy simulates the state, not the measurement record.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg meas[6];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
