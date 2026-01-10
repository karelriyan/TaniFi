// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract TaniVault {
    address public owner;
    uint256 public totalDeposited;

    struct Project {
        uint256 amount;
        uint256 timestamp;
    }

    mapping(uint256 => Project) public projects;
    uint256 public projectCount;

    event Deposit(address indexed investor, uint256 amount);
    event ProjectCreated(uint256 indexed projectId, uint256 amount);

    modifier onlyOwner() {
        _onlyOwner();
        _;
    }

    function _onlyOwner() internal view {
        require(msg.sender == owner, "Not authorized");
    }

    constructor() {
        owner = msg.sender;
    }

    function deposit() external payable {
        require(msg.value > 0, "Invalid amount");
        totalDeposited += msg.value;
        emit Deposit(msg.sender, msg.value);
    }

    function createProject(uint256 amount) external onlyOwner {
        require(amount > 0, "Invalid amount");

        projects[projectCount] = Project({
            amount: amount,
            timestamp: block.timestamp
        });

        emit ProjectCreated(projectCount, amount);
        projectCount++;
    }

    function getBalance() external view returns (uint256) {
        return address(this).balance;
    }
}
