use serde::Serialize;

use crate::Compiler;


impl Serialize for Compiler {
    fn serialize(&self, serializer: &mut Serializer) -> Result<(), Error> {
        let machine = target
            .create_target_machine(
                &triple,
                TargetMachine::get_host_cpu_name().to_string().as_str(),
                TargetMachine::get_host_cpu_features().to_string().as_str(),
                inkwell::OptimizationLevel::Default,
                inkwell::targets::RelocMode::Default,
                inkwell::targets::CodeModel::Default,
            )
            .unwrap();

        let buffer = target_machine.write_to_memory_buffer(&module, FileType::Assembly).unwrap();
        let slice = buffer.as_slice();


        let mut state = serializer.serialize_struct("Compiler", 3)?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("version", &self.version)?;
        state.serialize_field("target", &self.target)?;
        state.end()
    }
}