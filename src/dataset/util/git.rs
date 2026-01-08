use std::{ffi::OsStr, fs, path::Path, process::Command};

pub fn clone(url: &str, path: impl AsRef<Path>, include_lfs: bool) {
    let path = path.as_ref();
    println!("Cloning {url} to {path:?}");
    let cwd = path.parent().unwrap();
    let name = path.file_name().unwrap();
    fs::create_dir_all(cwd).unwrap();
    let mut cmd = Command::new("git");
    cmd.arg("clone").arg(url).arg(name);
    if !include_lfs {
        cmd.env("GIT_LFS_SKIP_SMUDGE", "1").current_dir(cwd);
    }
    let status = cmd.spawn().and_then(|mut child| child.wait()).unwrap();
    assert_eq!(status.code(), Some(0));
}

pub fn lfs_pull<S: AsRef<OsStr>>(path: impl AsRef<Path>, files: impl IntoIterator<Item = S>) {
    let path = path.as_ref();
    let files: Vec<S> = files.into_iter().collect();

    let lfs_list = Command::new("git")
        .arg("lfs")
        .arg("ls-files")
        .current_dir(path)
        .output()
        .unwrap();
    if lfs_list.status.code().unwrap() != 0 {
        panic!(
            "LFS list error\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&lfs_list.stdout),
            String::from_utf8_lossy(&lfs_list.stderr),
        );
    }
    let lfs_list = String::from_utf8(lfs_list.stdout).unwrap();
    let lfs_files: Vec<_> = lfs_list
        .lines()
        .map(|line| {
            let mut iter = line.split(' ');
            iter.next().unwrap();
            let st = iter.next().unwrap();
            let name = iter.next().unwrap();
            (OsStr::new(name), st == "*")
        })
        .collect();

    let mut cmd = Command::new("git");
    cmd.arg("lfs").arg("pull");
    let mut need_pull = false;
    for f in files {
        let pulled = lfs_files
            .iter()
            .find_map(|(name, pulled)| {
                if *name == f.as_ref() {
                    Some(pulled)
                } else {
                    None
                }
            })
            .expect("No such file in LFS");
        if !pulled {
            cmd.arg("-I").arg(f);
            need_pull = true;
        }
    }
    if need_pull {
        println!("Fetching files via LFS to {path:?}");
        let status = cmd
            .current_dir(path)
            .spawn()
            .and_then(|mut child| child.wait())
            .unwrap();
        assert_eq!(status.code(), Some(0));
    }
}
