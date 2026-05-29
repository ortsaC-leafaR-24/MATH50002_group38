{
  description = "Scientific Python shell for 2D strange attractor studies";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python3.withPackages (
            ps: with ps; [
              matplotlib
              numpy
              pandas
              scipy
            ]
          );
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              python
              ruff
              basedpyright
            ];

            shellHook = ''
              export MPLBACKEND=Agg
              export PYTHONNOUSERSITE=1
            '';
          };
        }
      );
    };
}
