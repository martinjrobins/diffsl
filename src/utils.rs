use std::env;
use std::path::Path;
use std::process::Command;

use anyhow::Result;
use anyhow::anyhow;

fn is_executable_on_path(executable_name: &str) -> bool {
    let output = Command::new("which")
        .arg(executable_name)
        .output()
        .expect("failed to execute which command");

    output.status.success()
}


pub fn find_executable<'a>(varients: &[&'a str]) -> Result<&'a str> {
    let mut command = None;
    for varient in varients {
        if is_executable_on_path(varient) {
            command = Some(varient.to_owned());
            break;
        }
    }
    match command {
        Some(command) => Ok(command),
        None => Err(anyhow!("Could not find any of {:?} on path", varients)),
    }
}


pub fn find_runtime_path(libraries: &[&str] ) -> Result<String> {
    // search in EMSDK lib dir and LIBRARY_PATH env variable
    let emsdk_lib = env::var("EMSDK").unwrap_or("".to_owned()) + "/upstream/emscripten/cache/sysroot/lib";
    let emsdk_lib_paths = vec![emsdk_lib.as_str()];
    let lib_path_env = env::var("LIBRARY_PATH").unwrap_or("".to_owned());
    let library_paths = lib_path_env.split(":").collect::<Vec<_>>();
    for &path in emsdk_lib_paths.iter().chain(library_paths.iter()) {
        // check if all librarys are in the path
        let mut found = true;
        for library in libraries {
            let library_path = Path::new(path).join(library);
            if !library_path.exists() {
                found = false;
                break;
            }
        }
        if found {
            return Ok(path.to_owned());
        }
    }
    Err(anyhow!("Could not find {:?} in LIBRARY_PATH {:?}", libraries, emsdk_lib_paths.iter().chain(library_paths.iter())))
}

pub fn find_library_path(varients: &[& str]) -> Result<String> {
    let library_paths_env = env::var("LIBRARY_PATH").unwrap_or("".to_owned());
    let library_paths = library_paths_env.split(":").collect::<Vec<_>>();
    for &path in library_paths.iter() {
        // check if one of the varients is in the path
        for varient in varients {
            let library_path = Path::new(path).join(varient);
            if library_path.exists() {
                let filename = library_path.as_os_str().to_str().unwrap().to_owned();
                return Ok(filename);
            }
        }
    }
    Err(anyhow!("Could not find any of {:?} in LIBRARY_PATH {:?}", varients, library_paths))
}