from pytorch_lightning.cli import LightningCLI

def main():
    LightningCLI(
        seed_everything_default=True,
        save_config_callback=None,  # keep minimal; remove if you want config saving
    )

if __name__ == "__main__":
    main()