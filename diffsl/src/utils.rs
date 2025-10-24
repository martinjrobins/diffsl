use std::env;
use std::path::Path;

use anyhow::anyhow;
use anyhow::Result;

fn is_executable_on_path(executable_name: &str) -> bool {
    for path in env::var("PATH").unwrap().split(':') {
        let path = Path::new(path).join(executable_name);
        if path.exists() {
            return true;
        }
    }
    false
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

pub fn find_runtime_path(libraries: &[&str]) -> Result<String> {
    // search in EMSDK lib dir and LIBRARY_PATH env variable
    let emsdk_lib =
        env::var("EMSDK").unwrap_or("".to_owned()) + "/upstream/emscripten/cache/sysroot/lib";
    let emsdk_lib_paths = vec![emsdk_lib.as_str()];
    let lib_path_env = env::var("LIBRARY_PATH").unwrap_or("".to_owned());
    let library_paths = lib_path_env.split(':').collect::<Vec<_>>();
    let all_paths = emsdk_lib_paths
        .into_iter()
        .chain(library_paths)
        .collect::<Vec<_>>();
    let mut failed_paths = Vec::new();
    for &path in all_paths.iter() {
        // check if all librarys are in the path
        let mut found = true;
        for library in libraries {
            let library_path = Path::new(path).join(library);
            if !library_path.exists() {
                failed_paths.push(library_path.as_os_str().to_str().unwrap().to_owned());
                found = false;
                break;
            }
        }
        if found {
            return Ok(path.to_owned());
        }
    }
    Err(anyhow!(
        "Could not find {:?} in LIBRARY_PATH {:?}, failed to find {:?}",
        libraries,
        all_paths,
        failed_paths
    ))
}

pub fn find_library_path(varients: &[&str]) -> Result<String> {
    let library_paths_env = env::var("LIBRARY_PATH").unwrap_or("".to_owned());
    let library_paths = library_paths_env.split(':').collect::<Vec<_>>();
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
    Err(anyhow!(
        "Could not find any of {:?} in LIBRARY_PATH {:?}",
        varients,
        library_paths
    ))
}
