// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Script, console} from "forge-std/Script.sol";
import {TaniVault} from "../src/TaniVault.sol";
import {MockIDRX} from "../src/MockIDRX.sol";
import {FarmerRegistry} from "../src/FarmerRegistry.sol";

/// @title TaniFi Deployment Script
/// @notice Foundry script that deploys all TaniFi contracts to Base Sepolia
/// @dev Run with: forge script script/Deploy.s.sol --rpc-url $BASE_RPC_URL --broadcast
contract Deploy is Script {
    function run() external {
        uint256 deployerKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerKey);

        console.log("Deploying TaniFi contracts...");
        console.log("Deployer address:", deployer);

        vm.startBroadcast(deployerKey);

        // 1. Deploy MockIDRX (Stablecoin)
        MockIDRX idrx = new MockIDRX();
        console.log("MockIDRX deployed at:", address(idrx));

        // 2. Deploy TaniVault with IDRX address and deployer as treasury
        TaniVault vault = new TaniVault(address(idrx), deployer);
        console.log("TaniVault deployed at:", address(vault));

        // 3. Deploy FarmerRegistry
        FarmerRegistry registry = new FarmerRegistry();
        console.log("FarmerRegistry deployed at:", address(registry));

        // 4. Configure FarmerRegistry with TaniVault address
        registry.setTaniVault(address(vault));
        console.log("FarmerRegistry configured with TaniVault");

        // 5. Add deployer as cooperative (for testing)
        vault.addCooperative(deployer);
        console.log("Deployer added as cooperative");

        // 6. Mint test tokens to deployer
        idrx.faucet(); // Gives deployer 10M IDRX
        console.log("Test IDRX minted to deployer");

        vm.stopBroadcast();

        console.log("");
        console.log("=== Deployment Complete ===");
        console.log("MockIDRX:       ", address(idrx));
        console.log("TaniVault:      ", address(vault));
        console.log("FarmerRegistry: ", address(registry));
        console.log("===========================");
    }
}

/// @title Deploy Individual Contracts
/// @notice Utility scripts for deploying contracts individually
contract DeployIDRX is Script {
    function run() external {
        uint256 deployerKey = vm.envUint("PRIVATE_KEY");
        vm.startBroadcast(deployerKey);
        MockIDRX idrx = new MockIDRX();
        console.log("MockIDRX deployed at:", address(idrx));
        vm.stopBroadcast();
    }
}

contract DeployVault is Script {
    function run() external {
        uint256 deployerKey = vm.envUint("PRIVATE_KEY");
        address idrxAddress = vm.envAddress("IDRX_ADDRESS");
        address treasury = vm.envAddress("TREASURY_ADDRESS");

        vm.startBroadcast(deployerKey);
        TaniVault vault = new TaniVault(idrxAddress, treasury);
        console.log("TaniVault deployed at:", address(vault));
        vm.stopBroadcast();
    }
}

contract DeployRegistry is Script {
    function run() external {
        uint256 deployerKey = vm.envUint("PRIVATE_KEY");
        vm.startBroadcast(deployerKey);
        FarmerRegistry registry = new FarmerRegistry();
        console.log("FarmerRegistry deployed at:", address(registry));
        vm.stopBroadcast();
    }
}
