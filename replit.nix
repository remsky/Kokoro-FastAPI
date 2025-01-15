{pkgs}: {
  deps = [
    pkgs.libsndfile
    pkgs.re2
    pkgs.oneDNN
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.rustc
    pkgs.libiconv
    pkgs.cargo
    pkgs.espeak-ng
    pkgs.libxcrypt
  ];
}
