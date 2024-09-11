{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
  in {
    devShells."${system}"."default" = let
      pkgs = import nixpkgs {
        inherit system;
      };
      pyEnv = pkgs.python3.withPackages (ps:
        with ps; [
          jupyter
          numpy
        ]);
    in
      pkgs.mkShell {
        packages = [
          pyEnv
        ];
        shellHook = ''

        '';
      };
  };
}
