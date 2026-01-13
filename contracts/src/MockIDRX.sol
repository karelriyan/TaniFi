// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {ERC20Burnable} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

/// @title MockIDRX - Mock Indonesian Rupiah Stablecoin
/// @notice Test token representing IDRX stablecoin for TaniFi development
/// @dev ERC20 implementation with 2 decimals (Rupiah convention) and minting capability
/// @author TaniFi Team - Lisk Builders Challenge 2026

contract MockIDRX is ERC20, ERC20Burnable, Ownable {
    // ============ State Variables ============

    mapping(address => bool) public minters;

    // ============ Events ============

    event MinterAdded(address indexed minter);
    event MinterRemoved(address indexed minter);

    // ============ Errors ============

    error NotMinter();
    error InvalidAddress();

    // ============ Modifiers ============

    modifier onlyMinter() {
        if (!minters[msg.sender] && msg.sender != owner()) revert NotMinter();
        _;
    }

    // ============ Constructor ============

    constructor() ERC20("Mock IDRX Stablecoin", "IDRX") Ownable(msg.sender) {
        minters[msg.sender] = true;
    }

    // ============ ERC20 Overrides ============

    /// @notice Returns 2 decimals (Rupiah convention - sen)
    function decimals() public pure override returns (uint8) {
        return 2;
    }

    // ============ Minting Functions ============

    /// @notice Mint new tokens (for testing purposes)
    /// @param to Recipient address
    /// @param amount Amount to mint (in smallest unit - sen)
    function mint(address to, uint256 amount) external onlyMinter {
        if (to == address(0)) revert InvalidAddress();
        _mint(to, amount);
    }

    // ============ Admin Functions ============

    function addMinter(address minter) external onlyOwner {
        if (minter == address(0)) revert InvalidAddress();
        minters[minter] = true;
        emit MinterAdded(minter);
    }

    function removeMinter(address minter) external onlyOwner {
        minters[minter] = false;
        emit MinterRemoved(minter);
    }

    // ============ Faucet Function (Testnet Only) ============

    /// @notice Faucet function for testing - anyone can get test tokens
    /// @dev Only for testnet deployment - remove in production
    function faucet() external {
        uint256 faucetAmount = 10_000_000 * (10 ** decimals()); // 10 million IDRX
        _mint(msg.sender, faucetAmount);
    }

    /// @notice Faucet with custom amount (minter only)
    /// @param to Recipient
    /// @param amount Amount in IDRX (will be converted to smallest unit)
    function faucetTo(address to, uint256 amount) external onlyMinter {
        if (to == address(0)) revert InvalidAddress();
        uint256 amountInSmallest = amount * (10 ** decimals());
        _mint(to, amountInSmallest);
    }
}
