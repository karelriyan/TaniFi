// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title MockIDRX - Mock Indonesian Rupiah Stablecoin
/// @notice Test token representing IDRX stablecoin for TaniFi development
/// @dev ERC20 implementation with 2 decimals (Rupiah convention) and minting capability

contract MockIDRX {
    // ============ State Variables ============

    string public constant name = "Mock IDRX Stablecoin";
    string public constant symbol = "IDRX";
    uint8 public constant decimals = 2; // 2 decimals for Rupiah (sen)

    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    address public owner;
    mapping(address => bool) public minters;

    // ============ Events ============

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event MinterAdded(address indexed minter);
    event MinterRemoved(address indexed minter);

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "IDRX: not owner");
        _;
    }

    modifier onlyMinter() {
        require(minters[msg.sender] || msg.sender == owner, "IDRX: not minter");
        _;
    }

    // ============ Constructor ============

    constructor() {
        owner = msg.sender;
        minters[msg.sender] = true;
    }

    // ============ ERC20 Functions ============

    function transfer(address to, uint256 amount) external returns (bool) {
        require(to != address(0), "IDRX: transfer to zero address");
        require(balanceOf[msg.sender] >= amount, "IDRX: insufficient balance");

        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;

        emit Transfer(msg.sender, to, amount);
        return true;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        require(spender != address(0), "IDRX: approve to zero address");

        allowance[msg.sender][spender] = amount;

        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) external returns (bool) {
        require(from != address(0), "IDRX: transfer from zero address");
        require(to != address(0), "IDRX: transfer to zero address");
        require(balanceOf[from] >= amount, "IDRX: insufficient balance");
        require(allowance[from][msg.sender] >= amount, "IDRX: insufficient allowance");

        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;

        emit Transfer(from, to, amount);
        return true;
    }

    // ============ Minting Functions ============

    /// @notice Mint new tokens (for testing purposes)
    /// @param to Recipient address
    /// @param amount Amount to mint (in smallest unit - sen)
    function mint(address to, uint256 amount) external onlyMinter {
        require(to != address(0), "IDRX: mint to zero address");

        totalSupply += amount;
        balanceOf[to] += amount;

        emit Transfer(address(0), to, amount);
    }

    /// @notice Burn tokens
    /// @param amount Amount to burn
    function burn(uint256 amount) external {
        require(balanceOf[msg.sender] >= amount, "IDRX: burn exceeds balance");

        balanceOf[msg.sender] -= amount;
        totalSupply -= amount;

        emit Transfer(msg.sender, address(0), amount);
    }

    /// @notice Burn tokens from an address (requires approval)
    /// @param from Address to burn from
    /// @param amount Amount to burn
    function burnFrom(address from, uint256 amount) external {
        require(balanceOf[from] >= amount, "IDRX: burn exceeds balance");
        require(allowance[from][msg.sender] >= amount, "IDRX: insufficient allowance");

        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        totalSupply -= amount;

        emit Transfer(from, address(0), amount);
    }

    // ============ Admin Functions ============

    function addMinter(address minter) external onlyOwner {
        require(minter != address(0), "IDRX: invalid minter");
        minters[minter] = true;
        emit MinterAdded(minter);
    }

    function removeMinter(address minter) external onlyOwner {
        minters[minter] = false;
        emit MinterRemoved(minter);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "IDRX: invalid owner");
        owner = newOwner;
    }

    // ============ Faucet Function (Testnet Only) ============

    /// @notice Faucet function for testing - anyone can get test tokens
    /// @dev Only for testnet deployment - remove in production
    function faucet() external {
        uint256 faucetAmount = 10_000_000 * (10 ** decimals); // 10 million IDRX
        totalSupply += faucetAmount;
        balanceOf[msg.sender] += faucetAmount;
        emit Transfer(address(0), msg.sender, faucetAmount);
    }

    /// @notice Faucet with custom amount (owner only)
    /// @param to Recipient
    /// @param amount Amount in IDRX (will be converted to smallest unit)
    function faucetTo(address to, uint256 amount) external onlyMinter {
        require(to != address(0), "IDRX: invalid recipient");
        uint256 amountInSmallest = amount * (10 ** decimals);
        totalSupply += amountInSmallest;
        balanceOf[to] += amountInSmallest;
        emit Transfer(address(0), to, amountInSmallest);
    }
}
