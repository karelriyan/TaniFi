// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";

/// @title TaniVault - Sharia-Compliant Agricultural Finance Protocol
/// @notice Core vault implementing Musyarakah (profit-sharing) for agricultural projects
/// @dev Manages pooling of investor funds, project lifecycle, and profit distribution
/// @author TaniFi Team - Base Indonesia Hackathon 2025

contract TaniVault is Ownable, ReentrancyGuard, Pausable {
    // ============ Enums ============

    enum ProjectState {
        FUNDRAISING,    // Project is accepting investments
        ACTIVE,         // Funds disbursed, farming in progress
        HARVESTED,      // Harvest completed, awaiting finalization
        FAILED,         // Project failed (natural disaster, etc.)
        COMPLETED       // Profit distributed, project closed
    }

    // ============ Structs ============

    struct Project {
        uint256 id;
        address farmer;           // Farmer's wallet (Smart Contract Wallet)
        address cooperative;      // Cooperative/Off-taker who validates
        address approvedVendor;   // Whitelisted vendor for input purchases
        uint256 targetAmount;     // Target funding amount (in IDRX wei)
        uint256 fundedAmount;     // Currently funded amount
        uint256 farmerShareBps;   // Farmer profit share (basis points, e.g., 3000 = 30%)
        uint256 investorShareBps; // Investor profit share (e.g., 7000 = 70%)
        uint256 startTime;        // When project was funded and started
        uint256 harvestTime;      // Expected harvest date
        uint256 harvestRevenue;   // Actual harvest revenue (set during finalization)
        ProjectState state;
        string ipfsMetadata;      // IPFS hash for project documentation
    }

    // ============ State Variables ============

    // Access Control
    mapping(address => bool) public cooperatives;
    mapping(address => bool) public operators;

    // Token
    IERC20 public stablecoin;  // IDRX token

    // Projects
    mapping(uint256 => Project) public projects;
    uint256 public projectCount;

    // Investments tracking: projectId => investor => amount
    mapping(uint256 => mapping(address => uint256)) public investments;

    // Investor returns: projectId => investor => claimable amount
    mapping(uint256 => mapping(address => uint256)) public investorReturns;

    // Total invested per project
    mapping(uint256 => uint256) public totalInvested;

    // Platform
    uint256 public constant PLATFORM_FEE_BPS = 100; // 1%
    uint256 public constant BPS_DENOMINATOR = 10000;
    address public treasury;

    // ============ Events ============

    event ProjectCreated(
        uint256 indexed projectId,
        address indexed farmer,
        address cooperative,
        uint256 targetAmount
    );

    event InvestmentReceived(
        uint256 indexed projectId,
        address indexed investor,
        uint256 amount
    );

    event FundsDisbursed(
        uint256 indexed projectId,
        address indexed vendor,
        uint256 amount
    );

    event HarvestReported(
        uint256 indexed projectId,
        uint256 revenue
    );

    event ProfitDistributed(
        uint256 indexed projectId,
        uint256 farmerShare,
        uint256 investorPool,
        uint256 platformFee
    );

    event InvestorWithdrawal(
        uint256 indexed projectId,
        address indexed investor,
        uint256 amount
    );

    event ProjectFailed(
        uint256 indexed projectId,
        string reason
    );

    event CooperativeAdded(address indexed cooperative);
    event CooperativeRemoved(address indexed cooperative);
    event OperatorAdded(address indexed operator);
    event OperatorRemoved(address indexed operator);
    event TreasuryUpdated(address indexed newTreasury);

    // ============ Errors ============

    error NotCooperative();
    error NotOperator();
    error NotAuthorized();
    error InvalidAddress();
    error InvalidAmount();
    error InvalidTarget();
    error FarmerShareTooHigh();
    error InvalidHarvestTime();
    error NotFundraising();
    error ExceedsTarget();
    error TransferFailed();
    error NotActive();
    error NoFunds();
    error WrongCooperative();
    error NotHarvested();
    error NoRevenue();
    error NotCompleted();
    error NoInvestment();
    error InvalidState();
    error NotFailed();

    // ============ Modifiers ============

    modifier onlyCooperative() {
        if (!cooperatives[msg.sender]) revert NotCooperative();
        _;
    }

    modifier onlyOperator() {
        if (!operators[msg.sender] && msg.sender != owner()) revert NotOperator();
        _;
    }

    // ============ Constructor ============

    constructor(address _stablecoin, address _treasury) Ownable(msg.sender) {
        if (_stablecoin == address(0)) revert InvalidAddress();
        if (_treasury == address(0)) revert InvalidAddress();

        stablecoin = IERC20(_stablecoin);
        treasury = _treasury;
        operators[msg.sender] = true;
    }

    // ============ Admin Functions ============

    function addCooperative(address _cooperative) external onlyOwner {
        if (_cooperative == address(0)) revert InvalidAddress();
        cooperatives[_cooperative] = true;
        emit CooperativeAdded(_cooperative);
    }

    function removeCooperative(address _cooperative) external onlyOwner {
        cooperatives[_cooperative] = false;
        emit CooperativeRemoved(_cooperative);
    }

    function addOperator(address _operator) external onlyOwner {
        if (_operator == address(0)) revert InvalidAddress();
        operators[_operator] = true;
        emit OperatorAdded(_operator);
    }

    function removeOperator(address _operator) external onlyOwner {
        operators[_operator] = false;
        emit OperatorRemoved(_operator);
    }

    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    function setTreasury(address _treasury) external onlyOwner {
        if (_treasury == address(0)) revert InvalidAddress();
        treasury = _treasury;
        emit TreasuryUpdated(_treasury);
    }

    // ============ Project Lifecycle Functions ============

    /// @notice Create a new agricultural project (called by cooperative)
    /// @param _farmer Address of the farmer's wallet
    /// @param _approvedVendor Whitelisted vendor for purchasing inputs
    /// @param _targetAmount Target funding amount in IDRX
    /// @param _farmerShareBps Farmer's profit share in basis points
    /// @param _harvestTime Expected harvest timestamp
    /// @param _ipfsMetadata IPFS hash containing project documentation
    function createProject(
        address _farmer,
        address _approvedVendor,
        uint256 _targetAmount,
        uint256 _farmerShareBps,
        uint256 _harvestTime,
        string calldata _ipfsMetadata
    ) external onlyCooperative whenNotPaused returns (uint256) {
        if (_farmer == address(0)) revert InvalidAddress();
        if (_approvedVendor == address(0)) revert InvalidAddress();
        if (_targetAmount == 0) revert InvalidTarget();
        if (_farmerShareBps > 5000) revert FarmerShareTooHigh(); // Max 50%
        if (_harvestTime <= block.timestamp) revert InvalidHarvestTime();

        uint256 projectId = projectCount;
        uint256 investorShareBps = BPS_DENOMINATOR - _farmerShareBps - PLATFORM_FEE_BPS;

        projects[projectId] = Project({
            id: projectId,
            farmer: _farmer,
            cooperative: msg.sender,
            approvedVendor: _approvedVendor,
            targetAmount: _targetAmount,
            fundedAmount: 0,
            farmerShareBps: _farmerShareBps,
            investorShareBps: investorShareBps,
            startTime: 0,
            harvestTime: _harvestTime,
            harvestRevenue: 0,
            state: ProjectState.FUNDRAISING,
            ipfsMetadata: _ipfsMetadata
        });

        projectCount++;

        emit ProjectCreated(projectId, _farmer, msg.sender, _targetAmount);
        return projectId;
    }

    /// @notice Invest in a project (public - any investor can call)
    /// @param _projectId ID of the project to invest in
    /// @param _amount Amount of IDRX to invest
    function invest(uint256 _projectId, uint256 _amount)
        external
        nonReentrant
        whenNotPaused
    {
        Project storage project = projects[_projectId];

        if (project.state != ProjectState.FUNDRAISING) revert NotFundraising();
        if (_amount == 0) revert InvalidAmount();
        if (project.fundedAmount + _amount > project.targetAmount) revert ExceedsTarget();

        // Transfer IDRX from investor to vault
        if (!stablecoin.transferFrom(msg.sender, address(this), _amount)) {
            revert TransferFailed();
        }

        // Update state
        project.fundedAmount += _amount;
        investments[_projectId][msg.sender] += _amount;
        totalInvested[_projectId] += _amount;

        emit InvestmentReceived(_projectId, msg.sender, _amount);

        // Auto-activate if fully funded
        if (project.fundedAmount == project.targetAmount) {
            _activateProject(_projectId);
        }
    }

    /// @dev Internal function to activate a fully funded project
    function _activateProject(uint256 _projectId) internal {
        Project storage project = projects[_projectId];
        project.state = ProjectState.ACTIVE;
        project.startTime = block.timestamp;
    }

    /// @notice Disburse funds to approved vendor (called by cooperative or operator)
    /// @param _projectId ID of the project
    function disburseToVendor(uint256 _projectId)
        external
        nonReentrant
        whenNotPaused
    {
        Project storage project = projects[_projectId];

        if (msg.sender != project.cooperative && !operators[msg.sender]) {
            revert NotAuthorized();
        }
        if (project.state != ProjectState.ACTIVE) revert NotActive();
        if (project.fundedAmount == 0) revert NoFunds();

        uint256 disbursementAmount = project.fundedAmount;

        // Transfer to approved vendor (closed-loop - not directly to farmer)
        if (!stablecoin.transfer(project.approvedVendor, disbursementAmount)) {
            revert TransferFailed();
        }

        emit FundsDisbursed(_projectId, project.approvedVendor, disbursementAmount);
    }

    /// @notice Report harvest revenue (called by cooperative)
    /// @param _projectId ID of the project
    /// @param _revenue Total revenue from harvest sale
    function reportHarvest(uint256 _projectId, uint256 _revenue)
        external
        onlyCooperative
    {
        Project storage project = projects[_projectId];

        if (project.cooperative != msg.sender) revert WrongCooperative();
        if (project.state != ProjectState.ACTIVE) revert NotActive();

        project.harvestRevenue = _revenue;
        project.state = ProjectState.HARVESTED;

        emit HarvestReported(_projectId, _revenue);
    }

    /// @notice Finalize harvest and distribute profits (Musyarakah settlement)
    /// @param _projectId ID of the project
    function finalizeHarvest(uint256 _projectId)
        external
        nonReentrant
        whenNotPaused
    {
        Project storage project = projects[_projectId];

        if (msg.sender != project.cooperative && !operators[msg.sender]) {
            revert NotAuthorized();
        }
        if (project.state != ProjectState.HARVESTED) revert NotHarvested();
        if (project.harvestRevenue == 0) revert NoRevenue();

        uint256 totalRevenue = project.harvestRevenue;

        // Calculate shares using Musyarakah profit-sharing
        uint256 platformFee = (totalRevenue * PLATFORM_FEE_BPS) / BPS_DENOMINATOR;
        uint256 farmerShare = (totalRevenue * project.farmerShareBps) / BPS_DENOMINATOR;
        uint256 investorPool = totalRevenue - farmerShare - platformFee;

        // Cooperative must deposit the revenue first
        if (!stablecoin.transferFrom(msg.sender, address(this), totalRevenue)) {
            revert TransferFailed();
        }

        // Transfer farmer share immediately
        if (!stablecoin.transfer(project.farmer, farmerShare)) {
            revert TransferFailed();
        }

        // Transfer platform fee to treasury
        if (!stablecoin.transfer(treasury, platformFee)) {
            revert TransferFailed();
        }

        // Calculate and store investor returns proportionally
        _distributeInvestorReturns(_projectId, investorPool);

        project.state = ProjectState.COMPLETED;

        emit ProfitDistributed(_projectId, farmerShare, investorPool, platformFee);
    }

    /// @dev Internal function to distribute returns to investors based on their share
    function _distributeInvestorReturns(uint256 _projectId, uint256 _investorPool) internal {
        // Note: In production, this would iterate through investors or use a merkle tree
        // For MVP, investors can claim their proportional share
        // Their return = (their investment / total invested) * investorPool
        // This is calculated at withdrawal time
    }

    /// @notice Withdraw investor returns after project completion
    /// @param _projectId ID of the completed project
    function withdrawReturns(uint256 _projectId)
        external
        nonReentrant
    {
        Project storage project = projects[_projectId];
        if (project.state != ProjectState.COMPLETED) revert NotCompleted();

        uint256 invested = investments[_projectId][msg.sender];
        if (invested == 0) revert NoInvestment();

        // Calculate proportional return
        uint256 totalRev = project.harvestRevenue;
        uint256 platformFee = (totalRev * PLATFORM_FEE_BPS) / BPS_DENOMINATOR;
        uint256 farmerShare = (totalRev * project.farmerShareBps) / BPS_DENOMINATOR;
        uint256 investorPool = totalRev - farmerShare - platformFee;

        uint256 returnAmount = (invested * investorPool) / totalInvested[_projectId];

        // Clear investment to prevent double withdrawal
        investments[_projectId][msg.sender] = 0;

        // Transfer returns
        if (!stablecoin.transfer(msg.sender, returnAmount)) {
            revert TransferFailed();
        }

        emit InvestorWithdrawal(_projectId, msg.sender, returnAmount);
    }

    /// @notice Mark project as failed (e.g., natural disaster)
    /// @param _projectId ID of the project
    /// @param _reason Reason for failure
    function markProjectFailed(uint256 _projectId, string calldata _reason)
        external
        onlyOperator
    {
        Project storage project = projects[_projectId];
        if (project.state != ProjectState.FUNDRAISING && project.state != ProjectState.ACTIVE) {
            revert InvalidState();
        }

        project.state = ProjectState.FAILED;

        emit ProjectFailed(_projectId, _reason);
    }

    /// @notice Refund investors if project fails during fundraising
    /// @param _projectId ID of the failed project
    function refundInvestors(uint256 _projectId)
        external
        nonReentrant
    {
        Project storage project = projects[_projectId];
        if (project.state != ProjectState.FAILED) revert NotFailed();

        uint256 invested = investments[_projectId][msg.sender];
        if (invested == 0) revert NoInvestment();

        // Clear investment
        investments[_projectId][msg.sender] = 0;
        project.fundedAmount -= invested;

        // Refund
        if (!stablecoin.transfer(msg.sender, invested)) {
            revert TransferFailed();
        }
    }

    // ============ View Functions ============

    function getProject(uint256 _projectId)
        external
        view
        returns (Project memory)
    {
        return projects[_projectId];
    }

    function getInvestment(uint256 _projectId, address _investor)
        external
        view
        returns (uint256)
    {
        return investments[_projectId][_investor];
    }

    function getVaultBalance() external view returns (uint256) {
        return stablecoin.balanceOf(address(this));
    }

    function isCooperative(address _addr) external view returns (bool) {
        return cooperatives[_addr];
    }

    function isOperator(address _addr) external view returns (bool) {
        return operators[_addr];
    }

    /// @notice Calculate expected returns for an investor
    /// @param _projectId ID of the project
    /// @param _investor Address of the investor
    /// @param _expectedRevenue Expected harvest revenue
    function calculateExpectedReturns(
        uint256 _projectId,
        address _investor,
        uint256 _expectedRevenue
    ) external view returns (uint256) {
        uint256 invested = investments[_projectId][_investor];
        if (invested == 0 || totalInvested[_projectId] == 0) return 0;

        uint256 platformFee = (_expectedRevenue * PLATFORM_FEE_BPS) / BPS_DENOMINATOR;
        Project storage project = projects[_projectId];
        uint256 farmerShare = (_expectedRevenue * project.farmerShareBps) / BPS_DENOMINATOR;
        uint256 investorPool = _expectedRevenue - farmerShare - platformFee;

        return (invested * investorPool) / totalInvested[_projectId];
    }
}
