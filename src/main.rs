#[cfg(not(unix))]
compile_error!("heracles supports Unix only (macOS and Linux)");

mod chunk;
mod cli;
mod config;
mod embed;
mod extract;
mod filter;
mod grep;
mod index;
mod interrupt;
mod list;
mod lock;
mod memory;
mod search;

use anyhow::Result;
use clap::Parser;

use cli::{Cli, Command};
use config::{Config, Home};
use lock::MemoryLock;
use memory::Memory;

fn main() -> Result<()> {
    let cli = Cli::parse();
    let home = Home::resolve()?;

    match cli.command {
        Command::Init { force } => {
            home.init(force)?;
            println!("Initialized {}", home.root().display());
        }
        Command::Config => {
            println!("# {}", home.config_file().display());
            print!(
                "{}",
                toml::to_string_pretty(&Config::load(&home)?.redacted())?
            );
        }
        Command::Index {
            path,
            filter,
            depth,
        } => {
            let _lock = MemoryLock::acquire(&home)?;
            let filter = filter.build()?;
            let config = Config::load(&home)?;
            let mut memory = Memory::load(&home, &config)?;
            let depth = depth.get();
            let flags = index::Flags::index(filter.covers_index(depth));
            index::run(&home, &config, &mut memory, &path, &filter, depth, flags)?;
        }
        Command::Search(args) => {
            // Lock is taken inside search::run: grep skips it; hybrid greps first, then locks.
            let config = Config::load(&home)?;
            search::run(&home, &config, &args)?;
        }
        Command::List(scope) => {
            let _lock = MemoryLock::acquire(&home)?;
            let config = Config::load(&home)?;
            list::run(&home, &config, &scope)?;
        }
    }

    Ok(())
}
